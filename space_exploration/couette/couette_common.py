"""Single source of truth for the Couette pressure->velocity problem.

Used by train_run.py / bo_loop.py / eval_run.py so the data, architecture,
loss and S3 helpers never drift between notebooks and scripts.

Data: /data1/dataToYouri/couette_dns_ml_f32.h5
  - per realization R<n>: u, v, w, p_total, each (T, X=24, Y=33, Z=24)
  - x,z periodic; y Chebyshev-Gauss-Lobatto in [-1, 1]
  - channels layout after stack(axis=1): [u, v, w, p_total]
"""
from __future__ import annotations

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
H5_PATH = "/data1/dataToYouri/couette_dns_ml_f32.h5"
KEYS = ('u', 'v', 'w', 'p_total')

# Splits (identical to the training notebooks)
ALL_REALS = ['R13', 'R14', 'R15', 'R18', 'R19', 'R21', 'R22', 'R23',
             'R26', 'R28', 'R3', 'R4', 'R5']
VAL_REALS = ["R18", "R5"]
TEST_REALS = ["R3", "R14"]  # final benchmark only -- never use for decisions

# Domain (periodic) lengths, in units of h (half-gap)
LX = 1.755 * np.pi
LZ = 1.2 * np.pi
NX, NY, NZ = 24, 33, 24

# S3 (SeaweedFS) checkpointing
S3_BUCKET = "simulations"
S3_ENDPOINT = "http://localhost:8333"
S3_MODEL_PREFIX = "flow_models/couette"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _extract_one(real: str) -> np.ndarray:
    with h5py.File(H5_PATH, 'r') as f:
        data = [np.array(f[real][k]) for k in KEYS]
    return np.stack(data, axis=1)


def _sample(real: str) -> np.ndarray:
    # R3/R4/R5 are denser (every h/U); subsample to ~300 snapshots like the rest.
    s = np.s_[:-1:5000 // 300] if real in ["R3", "R4", "R5"] else np.s_[:]
    return _extract_one(real)[s]


def load_all() -> np.ndarray:
    """Concatenated, all-realizations dataset: (T_total, 4, 24, 33, 24)."""
    return np.concatenate([_sample(r) for r in ALL_REALS], axis=0)


def load_splits() -> dict:
    """Return dict with train/val raw arrays (physical units) + coords.

    Normalization stats are computed over ALL data (as in training) so the
    split and the model agree.
    """
    per_real = {r: _sample(r) for r in ALL_REALS}
    all_data = np.concatenate(list(per_real.values()), axis=0)

    train_reals = [r for r in ALL_REALS if r not in VAL_REALS + TEST_REALS]
    train_npds = np.concatenate([per_real[r] for r in train_reals])
    val_npds = np.concatenate([per_real[r] for r in VAL_REALS])

    with h5py.File(H5_PATH, 'r') as f:
        y = np.array(f["y"])
        x = np.array(f["x"])
        z = np.array(f["z"])

    return dict(all_data=all_data, train=train_npds, val=val_npds,
                x=x, y=y, z=z)


# Normalization (over time, x, z -- axes 0,2,4)
NORM_AXES = (0, 2, 4)


def normalization_stats(all_data: np.ndarray):
    mean = all_data.mean(NORM_AXES, keepdims=True)
    std = all_data.std(NORM_AXES, keepdims=True) + 1e-10
    return mean, std


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean) / std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding="same"),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding="same"),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class M4(nn.Module):
    """Pressure -> velocity network with a width-parameterized lift.

    lift_width : channels produced by the lift conv
    c3d        : channels of the 3D conv blocks
    n_blocks   : number of 3D ConvBlocks
    widen      : if >0, insert a Conv2d(pre_lift->widen) block before the lift
    """
    D = 33
    XD = 24
    ZD = 24

    def __init__(self, lift_width=32, c3d=32, n_blocks=5, widen=0):
        super().__init__()
        self.c3d = c3d
        self.pressure_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding="same"), nn.SiLU(),
            nn.Conv2d(8, 32, 3, padding="same"), nn.SiLU(),
        )
        in_lift = 32
        if widen > 0:
            self.pre_widen = nn.Sequential(
                nn.Conv2d(32, widen, 3, padding="same"), nn.SiLU())
            in_lift = widen
        self.lift = nn.Conv2d(in_lift, self.D * lift_width, 3, padding="same")
        self.lift_proj = (nn.Conv3d(lift_width, c3d, 1)
                          if lift_width != c3d else nn.Identity())
        self.conv_blocks = nn.Sequential(*[ConvBlock(c3d, c3d)
                                           for _ in range(n_blocks)])
        self.output = nn.Conv3d(c3d, 3, 3, padding="same")

    def forward(self, pressure):
        x = self.pressure_encoder(pressure)
        if hasattr(self, "pre_widen"):
            x = self.pre_widen(x)
        x = self.lift(x)
        x = x.reshape(-1, self.lift.out_channels // self.D, self.XD, self.D, self.ZD)
        x = self.lift_proj(x)
        x = self.conv_blocks(x)
        x = self.output(x)
        return x


def make_model(cfg: dict, device: str) -> M4:
    m = M4(lift_width=cfg.get("lift_width", 32),
           c3d=cfg.get("c3d", 32),
           n_blocks=cfg.get("n_blocks", 5),
           widen=cfg.get("widen", 0)).to(device)
    return m


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Loss (MSE + weighted spectral, identical to framework_2.ipynb)
# ---------------------------------------------------------------------------
class SpectralLoss(nn.Module):
    """L = MSE(pred, tgt) + lambda * sum_k W[k]|U_k - V_k|^2 / sum_k W[k]

    W[k] fixed per log-band, w_b = 1/(N_b * E_b), E_b from the TARGET spectrum.
    """

    def __init__(self, target, dim=(-2, -1), n_bands=6, kmin=1.0, lambda_weight=1.0):
        super().__init__()
        assert lambda_weight >= 0
        self.dim = dim
        self.lambda_weight = lambda_weight
        self.W = self._build_weights(target, dim, n_bands, kmin)

    def _build_weights(self, target, dim, n_bands, kmin):
        with torch.no_grad():
            target = target.detach()
            spec = torch.fft.fft2(target, dim=dim, norm="ortho")
            E = (spec.abs() ** 2).mean(dim=(0, 1, 3))            # [X, Z]
            Xn, Zn = E.shape
            kx = torch.fft.fftfreq(Xn, device=E.device) * (2 * np.pi)
            kz = torch.fft.fftfreq(Zn, device=E.device) * (2 * np.pi)
            KX, KZ = torch.meshgrid(kx, kz, indexing="ij")
            k_mag = torch.hypot(KX, KZ)
            k_max = k_mag.max()
            edges = torch.logspace(torch.log10(torch.tensor(kmin)),
                                   torch.log10(k_max), n_bands + 1,
                                   device=E.device)
            W = torch.zeros_like(k_mag)
            for b in range(n_bands):
                lo, hi = float(edges[b]), float(edges[b + 1])
                mask = (k_mag >= lo) & (k_mag < hi)
                N_b = max(int(mask.sum()), 1)
                E_b = E[mask].mean().clamp(min=1e-12)
                W[mask] = 1.0 / (N_b * E_b)
            W[k_mag == 0] = 0.0
            return W[None, None, :, None, :]

    def forward(self, pred, target):
        sp = torch.fft.fft2(pred, dim=self.dim, norm="ortho")
        st = torch.fft.fft2(target, dim=self.dim, norm="ortho")
        diff2 = (sp - st).abs() ** 2
        weighted = (diff2 * self.W).sum() / self.W.sum()
        mse = torch.nn.functional.mse_loss(pred, target)
        return mse + self.lambda_weight * weighted, mse, weighted


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CouetteDS(Dataset):
    """x = pressure (channel 3) at Y=0 level -> (N,1,X,Z);
    y = velocity (channels 0-2) -> (N,3,X,Y,Z)."""

    def __init__(self, raw: np.ndarray, mean, std, device: str):
        d = normalize(raw, mean, std)
        self.x = torch.from_numpy(d[:, 3:, :, 0, :]).float().to(device)
        self.y = torch.from_numpy(d[:, :3, :, :, :]).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def make_dataloaders(splits: dict, mean, std, device: str,
                     batch_size: int = 4):
    train_ds = CouetteDS(splits["train"], mean, std, device)
    val_ds = CouetteDS(splits["val"], mean, std, device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        drop_last=True)
    return train_dl, val_dl, train_ds, val_ds


# ---------------------------------------------------------------------------
# S3 checkpointing (save best only + load)
# ---------------------------------------------------------------------------
def _s3fs():
    import s3fs
    return s3fs.S3FileSystem(profile='default',
                             client_kwargs={'endpoint_url': S3_ENDPOINT})


def checkpoint_key(prefix: str, tag: str) -> str:
    return f"{S3_BUCKET}/{prefix}/{tag}.pt"


def save_model_to_s3(model: nn.Module, prefix: str, tag: str,
                     extra_cfg: dict | None = None, widen: int = 0):
    """Save weights + config to SeaweedFS S3."""
    import tempfile, os
    payload = {
        'model_state_dict': model.state_dict(),
        'config': {
            'lift_width': model.lift.out_channels // model.D,
            'c3d': model.c3d,
            'n_blocks': len(model.conv_blocks),
            'widen': int(widen),   # store the int so we can rebuild exactly
        },
    }
    if extra_cfg:
        payload['config'].update({k: (bool(v) if isinstance(v, bool) else v)
                                  for k, v in extra_cfg.items()})
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(payload, tmp)
        lpath = tmp.name
    key = checkpoint_key(prefix, tag)
    _s3fs().put_file(lpath=lpath, rpath=key)
    os.unlink(lpath)
    return key


def delete_s3(prefix: str, tag: str):
    import os as _os
    key = checkpoint_key(prefix, tag)
    try:
        _s3fs().rm(key)
        return True
    except Exception:
        return False


def load_model_from_s3(prefix: str, tag: str, device: str) -> tuple[M4, dict]:
    """Load a checkpoint, rebuild the architecture, return (model, config).

    If the stored config lacks 'widen', infer it from the state_dict shapes
    (pre_widen.0.weight = (widen, 32, 3, 3)).
    """
    import io
    buf = _s3fs().cat_file(checkpoint_key(prefix, tag))
    ckpt = torch.load(io.BytesIO(buf), map_location="cpu")
    cfg = dict(ckpt["config"])
    sd = ckpt["model_state_dict"]

    if "widen" in cfg and cfg["widen"]:
        widen = int(cfg["widen"])
    elif "pre_widen.0.weight" in sd:
        # pre_widen: Conv2d(32, widen) -> weight shape (widen, 32, 3, 3)
        widen = int(sd["pre_widen.0.weight"].shape[0])
    else:
        widen = 0
    cfg["widen"] = widen

    model = M4(lift_width=cfg["lift_width"], c3d=cfg["c3d"],
               n_blocks=cfg["n_blocks"], widen=widen).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def predict_phys(model: nn.Module, ds: Dataset, device: str,
                 mean_gpu, std_gpu) -> np.ndarray:
    """Run model over a dataset, unnormalize velocity -> (N,3,X,Y,Z) float32."""
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in ds:
            out.append(model(xb))
    pred_norm = torch.cat(out, dim=0)
    pred_phys = (pred_norm * std_gpu[:, :3]) + mean_gpu[:, :3]
    return pred_phys.cpu().numpy()
