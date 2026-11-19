"""Train one Channel-flow (A1) config and keep ONLY the best validation checkpoint.

Standalone (no notebook state). Mirrors the `full_2.ipynb` framework (DataNet,
dask-backed zarr S3 data, plain MSE loss) but runs as a script, and is inspired
by `space_exploration/couette/train_run.py`:

  1. As the BO objective:   from train_channel import run_trial; run_trial(params)
  2. From the CLI:          python train_channel.py --params '{"c":128,"n_blocks":6}'

The objective returned is the best val MSE (lower is better).

Differences from `full_2.ipynb`:
  - The whole dataset is assumed to fit in RAM (loaded once up front), so we
    build an in-memory dataset rather than re-opening zarr per worker.
  - An adaptive learning rate (ReduceLROnPlateau on val MSE) is used instead of
    a fixed LR.
"""
from __future__ import annotations

import argparse
import json
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import zarr
import s3fs

# ---------------------------------------------------------------------------
# S3 / dataset location (matches full_2.ipynb)
# ---------------------------------------------------------------------------
S3_BUCKET = "simulations"
STORE_PATH = "flow_export/A1"
S3_ENDPOINT = "http://localhost:8333"
S3_MODEL_PREFIX = "flow_models/channel"

# Fixed hyper-parameters (not part of the BO search)
EPOCHS = 20
BATCH_SIZE = 4
NUM_WORKERS = 16
SCHEDULER_PATIENCE = 3      # epochs without val improvement before dropping LR
SCHEDULER_FACTOR = 0.5      # LR multiplier on plateau
SCHEDULER_MIN_LR = 1e-6


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _s3fs():
    return s3fs.S3FileSystem(
        profile="default",
        client_kwargs={"endpoint_url": S3_ENDPOINT},
    )


def _open_data_zarr():
    return zarr.open(
        store=s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/data.zarr", s3=_s3fs()),
        mode="r",
    )


def _open_channel_zarr():
    return zarr.open(
        store=s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/channel.zarr", s3=_s3fs()),
        mode="r",
    )


def _range_to_slice(r: range) -> slice:
    """Convert a range to the equivalent slice (zarr accepts slices, not ranges)."""
    return slice(r.start, r.stop, r.step)


def compute_splits(T: int,
                   train_pc: int = 80, val_pc: int = 10, test_pc: int = 10,
                   spacing: int = 10, start_offset: int = 60,
                   train_stride: int = 1) -> dict:
    """Time split, ordered train -> val -> test along the time axis.

    Layout (train first, then val, then test):
      train : range(start_offset,
                    start_offset + train_size, train_stride)        (default contiguous)
      val   : contiguous block after train (+ spacing gap)
      test  : the remaining tail after val (+ spacing gap)

    Default `train_stride = 1` (contiguous train). Bump it to subsample.

    Returns a dict of **range** objects (support `len()` and iteration).
    Zarr indexing is done via `_range_to_slice`.
    """
    split_total = train_pc + val_pc + test_pc
    available_T = (T - spacing * 2 - start_offset)
    train_size = (available_T * train_pc) // split_total
    val_size = (available_T * val_pc) // split_total
    test_size = (available_T * test_pc) // split_total

    train_range = range(
        start_offset,
        start_offset + train_size,
        train_stride,
    )
    # val: contiguous block after train (+ spacing gap)
    train_last = max(train_range)
    val_start = train_last + spacing
    val_range = range(val_start, val_start + val_size, 1)
    # test: contiguous tail after val (+ spacing gap)
    test_start = val_start + val_size + spacing
    test_range = range(test_start, test_start + test_size, 1)

    return {"train": train_range, "val": val_range, "test": test_range}


def load_into_ram(z, idx, in_ch: int, out_ch: int):
    """Pull the selected timesteps into RAM once. Returns (x, y) float32 tensors.

    `idx` may be a `range` (zarr doesn't accept ranges, so it is converted to a
    slice before indexing) or already a slice / integer index array.

    Input  x: channel index slice 4:5 -> [T, 1, X, Y, Z] (wall pressure, 1 channel)
    Output y: channel slice :3      -> [T, 3, X, Y, Z]  (velocity, 3 channels)
    """
    if isinstance(idx, range):
        idx = _range_to_slice(idx)
    x = torch.from_numpy(np.asarray(z[idx, in_ch:in_ch + 1]).astype(np.float32))
    y = torch.from_numpy(np.asarray(z[idx, 0:out_ch]).astype(np.float32))
    return x, y


class RAMDataset(Dataset):
    """In-memory dataset: x (wall pressure), y (velocity)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _channel_stats(t: torch.Tensor) -> tuple:
    """Per-channel mean/std of a [T, C, X, Y, Z] tensor, over all non-channel dims."""
    td = t.double()
    dims = tuple(a for a in range(td.ndim) if a != 1)
    mean = td.mean(dim=dims).float().detach()  # [C]
    std = td.std(dim=dims).float().detach().clamp_min(1e-6)  # [C]
    return mean, std


def make_dataloaders(x: torch.Tensor, y: torch.Tensor,
                     x_mean: torch.Tensor, y_mean: torch.Tensor,
                     x_std: torch.Tensor, y_std: torch.Tensor,
                     batch_size: int, num_workers: int, device: str):
    """Build a normalized-batches generator + dataloader.

    Normalization is applied on-device (in place) to avoid extra CPU copies,
    mirroring the `normalized_batches` helper in full_2.ipynb.
    """
    ds = RAMDataset(x, y)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    y_mean_gpu = y_mean.to(device)
    y_std_gpu = y_std.to(device)
    x_mean_gpu = x_mean.to(device)
    x_std_gpu = x_std.to(device)

    def normalized_batches(dl_):
        for xb, yb in dl_:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yb.sub_(y_mean_gpu[None, :, None, None, None]) \
               .div_(y_std_gpu[None, :, None, None, None])
            xb.sub_(x_mean_gpu[None, :, None, None, None]) \
               .div_(x_std_gpu[None, :, None, None, None])
            yield xb, yb

    return dl, normalized_batches


# ---------------------------------------------------------------------------
# Model (full_2.ipynb DataNet)
# ---------------------------------------------------------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding="same"),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding="same"),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class DataNet(nn.Module):
    def __init__(self, c: int = 64, n_blocks: int = 3, dy: int = 129):
        super().__init__()
        self.dy = dy
        self.c = c
        self.core = nn.Sequential(
            nn.Conv3d(1, c, 3, padding="same"),
            *[ConvBlock3D(c) for _ in range(n_blocks)],
            nn.Conv3d(c, 3, 3, padding="same"),
        )

    def forward(self, p):
        x = p.expand(-1, -1, -1, self.dy, -1)
        return self.core(x)


def make_model(arch: dict, device: str) -> DataNet:
    model = DataNet(
        c=int(arch.get("c", 64)),
        n_blocks=int(arch.get("n_blocks", 3)),
        dy=int(arch.get("dy", 129)),
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# S3 checkpoint helpers (best-only, like train_run.py)
# ---------------------------------------------------------------------------
def _checkpoint_key(prefix: str, tag: str) -> str:
    return f"{prefix}/ckpt_{tag}.pt"


def save_model_to_s3(model: nn.Module, prefix: str, tag: str,
                     extra_cfg: dict | None = None):
    fs = _s3fs()
    payload = {
        "model_state_dict": model.state_dict(),
        "config": dict(extra_cfg or {}),
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        torch.save(payload, fh.name)
        remote = _checkpoint_key(prefix, tag)
        with fs.open(remote, "wb") as out:
            with open(fh.name, "rb") as src:
                out.write(src.read())


def delete_s3(prefix: str, tag: str):
    fs = _s3fs()
    key = _checkpoint_key(prefix, tag)
    try:
        fs.rm(key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def _val_mse(model, normalized_batches, val_dl, loss_fn) -> float:
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x, y in normalized_batches(val_dl):
            tot += loss_fn(model(x), y).item()
    return tot / len(val_dl)


def _run_epoch(model, opti, loss_fn, normalized_batches, train_dl):
    model.train()
    tot = 0.0
    for x, y in normalized_batches(train_dl):
        opti.zero_grad()
        l = loss_fn(model(x), y)
        l.backward()
        opti.step()
        tot += l.item()
    return tot / len(train_dl)


def _trackio_log(trial_label: str, epoch: int, train_err: float, val_mse: float,
                 arch: dict, best_so_far: float, lr_now: float):
    """Log a training step to a Trackio dashboard (best-effort)."""
    try:
        import trackio
        trackio.log({
            "epoch": epoch,
            "train/mse": train_err,
            "val/mse": val_mse,
            "val/EV": 1.0 - val_mse,
            "val/best_so_far": best_so_far,
            "opt/lr": lr_now,
            "arch/c": arch["c"],
            "arch/n_blocks": arch["n_blocks"],
            "arch/dy": arch["dy"],
            "trial": trial_label,
        })
    except Exception:
        pass  # tracking is optional; never let it break training


def run_trial(params: dict, *, device: str | None = None,
              log: bool = False, epochs: int | None = None,
              lr: float | None = None,
              trackio_project: str | None = None,
              trial_label: str | None = None) -> dict:
    """Train one config; return dict with 'val_mse' (objective) + metadata.

    Only the BEST (lowest val-MSE) checkpoint is kept on S3.
    Adaptive LR: ReduceLROnPlateau steps on val MSE.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or EPOCHS
    lr = lr or 1e-3
    trial_label = trial_label or "trial"

    arch = {
        "c": int(params.get("c", 64)),
        "n_blocks": int(params.get("n_blocks", 3)),
        "dy": int(params.get("dy", 129)),
    }

    if log:
        print(f"[trial] {arch}  lr={lr}  epochs={epochs}", flush=True)

    _trackio_ok = False
    if trackio_project is not None:
        try:
            import trackio
            trackio.init(project=trackio_project,
                         config={"arch": arch, "epochs": epochs, "lr": lr,
                                 "scheduler": "ReduceLROnPlateau",
                                 "patience": SCHEDULER_PATIENCE,
                                 "factor": SCHEDULER_FACTOR})
            _trackio_ok = True
        except Exception as e:
            if log:
                print(f"  [warn] trackio init failed: {e}", flush=True)

    # --- data (loaded once into RAM; assumed to fit) ---
    z = _open_data_zarr()
    T = z.shape[0]
    splits = compute_splits(T)
    if log:
        print(f"  T={T}  train={len(splits['train'])}  val={len(splits['val'])}",
              flush=True)

    # load train/val into RAM (input ch 4:5, output ch 0:3)
    xtr, ytr = load_into_ram(z, splits["train"], in_ch=4, out_ch=3)
    xva, yva = load_into_ram(z, splits["val"], in_ch=4, out_ch=3)

    # per-channel stats: y has 3 velocity channels, x has 1 pressure channel
    y_mean, y_std = _channel_stats(ytr)   # [3]
    x_mean, x_std = _channel_stats(xtr)   # [1]

    train_dl, train_batches = make_dataloaders(
        xtr, ytr, x_mean, y_mean, x_std, y_std,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device)
    val_dl, val_batches = make_dataloaders(
        xva, yva, x_mean, y_mean, x_std, y_std,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device)

    # free the raw zarr handle (data is in RAM now)
    del z

    model = make_model(arch, device)
    n_params = sum(p.numel() for p in model.parameters())
    if log:
        print(f"  model: {n_params/1e6:.2f}M params", flush=True)

    loss_fn = nn.MSELoss(reduction="mean")
    opti = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opti, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR,
    )

    prefix = f"{S3_MODEL_PREFIX}/c{arch['c']}_b{arch['n_blocks']}_dy{arch['dy']}"

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        train_err = _run_epoch(model, opti, loss_fn, train_batches, train_dl)
        val_mse = _val_mse(model, val_batches, val_dl, loss_fn)
        scheduler.step(val_mse)
        lr_now = opti.param_groups[0]["lr"]

        if log:
            print(f"  epoch {epoch:3d}  train={train_err:.4f}  val={val_mse:.4f}"
                  f"  lr={lr_now:.2e}", flush=True)
        if _trackio_ok:
            _trackio_log(trial_label, epoch, train_err, val_mse, arch,
                         best_val, lr_now)

        if val_mse < best_val:
            if best_epoch >= 0:
                delete_s3(prefix, f"best_{best_epoch:03d}")
            best_val = val_mse
            best_epoch = epoch
            save_model_to_s3(model, prefix, f"best_{epoch:03d}",
                             extra_cfg={"arch": arch, "epochs": epochs,
                                       "lr": lr})
            save_model_to_s3(model, prefix, "best",
                             extra_cfg={"arch": arch, "epochs": epochs,
                                       "lr": lr, "best_epoch": epoch})

    if _trackio_ok:
        try:
            import trackio
            trackio.finish()
        except Exception:
            pass

    return {
        "val_mse": best_val,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "arch": arch,
        "epochs": epochs,
        "lr": lr,
        "s3_prefix": prefix,
        "s3_best_tag": "best",
        "model": model,
    }


def _parse_params(s: str | None, defaults: dict | None = None) -> dict:
    d = dict(defaults or {})
    if s:
        d.update(json.loads(s))
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train one Channel-flow (A1) config")
    ap.add_argument("--params", type=str, default=None,
                    help='JSON arch params, e.g. \'{"c":128,"n_blocks":6}\'')
    ap.add_argument("--epochs", type=int, default=None,
                    help="override EPOCHS (default %d); use 2 for a smoke test" % EPOCHS)
    ap.add_argument("--lr", type=float, default=None,
                    help="initial learning rate (default 1e-3)")
    ap.add_argument("--project", type=str, default=None,
                    help="trackio project name (default: none)")
    ap.add_argument("--label", type=str, default="trial",
                    help="trial label for trackio")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    result = run_trial(
        _parse_params(args.params),
        epochs=args.epochs,
        lr=args.lr,
        trackio_project=args.project,
        trial_label=args.label,
        log=not args.quiet,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "model"},
                     indent=2))
