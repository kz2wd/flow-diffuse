#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Running")


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import trackio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import zarr
import s3fs


S3_BUCKET = "simulations"
STORE_PATH = "flow_export/A1"

fs = s3fs.S3FileSystem(profile='default', client_kwargs={
        'endpoint_url': 'http://localhost:9000'
    })

s3_map = s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/data.zarr", s3=fs)
z = zarr.open(
    store=s3_map,
    mode="r",
)
data_selection = z[60:-10:4]
all_data = np.concatenate((data_selection[:, :3], data_selection[:, 4:]), axis=1)  
T, C, X, Y, Z = all_data.shape
print("all_data", all_data.shape)

train_pc = 80
val_pc = 10
test_pc = 10
split_total = train_pc + val_pc + test_pc
# ignored indices between sets
spacing = 10
train_size = (T - spacing * 2 * train_pc) // split_total
val_size = (T - spacing * 2 * val_pc) // split_total
test_size = (T - spacing * 2 * test_pc) // split_total


val_npds = all_data[:val_size]
train_npds = all_data[val_size + spacing:train_size + spacing]

device = "cuda"

class DS(Dataset):
    def __init__(self, d):
        # input : wall pressure at y=0  -> [T, 1, X, Z]
        # target: full velocity        -> [T, 3, X, Y, Z]
        self.x = torch.from_numpy(d[:, 3:, :, 0, :]).float().to(device)
        self.y = torch.from_numpy(d[:, :3, :, :, :]).float().to(device)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, i):
        return self.x[i], self.y[i]

# physical grid (non-uniform in y) - needed for correct derivatives/divergence
def _domain():
    store = s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/channel.zarr", s3=fs)
    channel_z = zarr.open(
        store=store,
        mode="r",
    )
    return{'x': channel_z['x'][:], 'y': channel_z['y'][:], 'z': channel_z['z'][:]}
DOM = _domain()
x_grid, y_grid, z_grid = DOM['x'], DOM['y'], DOM['z']
dx = float(x_grid[1] - x_grid[0])          # uniform
dz = float(z_grid[1] - z_grid[0])          # uniform
dy = np.diff(y_grid)                        # non-uniform (wall-clustered)
print(f"grid  dx={dx:.5f}  dz={dz:.5f}  dy[min,max]={dy.min():.5f},{dy.max():.5f}  (y non-uniform)")

normalization_axes = (0, 2, 4,)
all_data_mean = all_data.mean(normalization_axes, keepdims=True)
all_data_std = all_data.std(normalization_axes, keepdims=True) + 1e-10
all_data_mean_gpu = torch.from_numpy(all_data_mean).to(device)
all_data_std_gpu = torch.from_numpy(all_data_std).to(device)

def normalize(data):
    return (data - all_data_mean) / all_data_std

def unnormalize_velocity_gpu(data):
    return (data * all_data_std_gpu[:, :3]) + all_data_mean_gpu[:, :3]

val_ds = DS(normalize(val_npds))
train_ds = DS(normalize(train_npds))

def predict(model, flat_x_ds):
    model.eval()
    with torch.no_grad():
        return model(flat_x_ds)

batch_size = 4
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
print("train", len(train_dl), "val", len(val_dl))


# In[9]:


print(all_data.shape)


# In[10]:


# ---------------------------------------------------------------------------
# PLACEHOLDER DataNet  (to be redesigned - see discussion)
#   wall pressure [B,1,X,Z]  ->  full velocity [B,3,X,Y,Z]
#   conv3d from the start: lift pressure to all depths, then 3D conv blocks.
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
    def __init__(self, c=32, n_blocks=4, D=129, xd=128, zd=128):
        super().__init__()
        self.D, self.xd, self.zd = D, xd, zd
        # 2D encoder on the wall pressure
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding="same"), nn.SiLU(),
            nn.Conv2d(16, c, 3, padding="same"), nn.SiLU(),
        )
        # lift: 2D [B,c,X,Z] -> 3D [B,c,X,D,Z]  (broadcast across depth)
        self.lift = nn.Conv2d(c, c * D, 3, padding="same")
        self.blocks = nn.Sequential(*[ConvBlock3D(c) for _ in range(n_blocks)])
        self.out = nn.Conv3d(c, 3, 3, padding="same")

    def forward(self, p):
        x = self.enc(p)                                  # [B,c,X,Z]
        x = self.lift(x)                                 # [B,c*D,X,Z]
        # lift.out_channels == c*D  ->  [B,c,X,D,Z]
        x = x.reshape(-1, x.shape[1] // self.D, self.xd, self.D, self.zd)
        x = self.blocks(x)                               # [B,c,X,D,Z]
        return self.out(x)                               # [B,3,X,D,Z]


def make_data_net(c=32, n_blocks=4):
    m = DataNet(c=c, n_blocks=n_blocks).to(device)
    n = sum(p.numel() for p in m.parameters())
    print(f"DataNet c={c} n_blocks={n_blocks} -> {n/1e6:.2f}M params")
    return m


# In[ ]:


# ---------------------------------------------------------------------------
# Loss  (stage 1: data-based initial estimate -> plain MSE on the full field)
#   The physics-correction loss (residual-based) is added in a later cell
#   once the correction architecture is built.
# ---------------------------------------------------------------------------
mse_loss = nn.MSELoss(reduction="mean")


# In[ ]:


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
epochs = 20
lr = 5e-4
optim = torch.optim.Adam
batch_size = 4
project_name = "full1"          # constant study name; runs distinguished by config

# --- DataNet (stage-1 initial estimate) knobs ---
net_c = 128
net_n_blocks = 5

model = make_data_net(c=net_c, n_blocks=net_n_blocks)

# fixed band edges for the per-band skill diagnostic
XN, ZN = 128, 128
_kx = torch.fft.fftfreq(XN) * (2 * np.pi)
_kz = torch.fft.fftfreq(ZN) * (2 * np.pi)
_KX, _KZ = torch.meshgrid(_kx, _kz, indexing="ij")
_KMAG = torch.hypot(_KX, _KZ)
BAND_EDGES = torch.logspace(torch.log10(torch.tensor(1.0)),
                            torch.log10(_KMAG.max()), 7).tolist()
print("bands:", [f"{BAND_EDGES[i]:.2f}-{BAND_EDGES[i+1]:.2f}" for i in range(6)])


# In[ ]:


# ---------------------------------------------------------------------------
# Train / eval / diagnostics
# ---------------------------------------------------------------------------
from tqdm import tqdm

def train(model, opti, loss_fn, ds):
    model.train()
    tot = 0.0
    for x, y in tqdm(ds):
        opti.zero_grad()
        l = loss_fn(model(x), y)
        l.backward()
        opti.step()
        tot += l.item()
    return tot / len(ds)


@torch.no_grad()
def evaluate(model, ds, loss_fn):
    model.eval()
    tot = 0.0
    for x, y in tqdm(ds):
        tot += loss_fn(model(x), y).item()
    return tot / len(ds)


@torch.no_grad()
def val_band_skill(model, ds, edges=BAND_EDGES, dim=(-2, -1)):
    """Per-band skill = 1 - E_err(b)/E_target(b) over x,z, full val set."""
    model.eval()
    Xn, Zn = ds.dataset.y.shape[2], ds.dataset.y.shape[4]
    kx = torch.fft.fftfreq(Xn, device=device) * (2 * np.pi)
    kz = torch.fft.fftfreq(Zn, device=device) * (2 * np.pi)
    KX, KZ = torch.meshgrid(kx, kz, indexing="ij")
    k_mag = torch.hypot(KX, KZ).to(device)
    band_err = [0.0] * (len(edges) - 1)
    band_tgt = [0.0] * (len(edges) - 1)
    for x, y in ds:
        pred = model(x)
        E_err = (torch.fft.fft2(pred - y, dim=dim, norm="ortho").abs() ** 2).mean(dim=(0, 1, 3))
        E_tgt = (torch.fft.fft2(y, dim=dim, norm="ortho").abs() ** 2).mean(dim=(0, 1, 3))
        for b in range(len(edges) - 1):
            mask = (k_mag >= edges[b]) & (k_mag < edges[b + 1])
            band_err[b] += float(E_err[mask].sum())
            band_tgt[b] += float(E_tgt[mask].sum())
    return {f"{edges[b]:.2f}-{edges[b+1]:.2f}": 1.0 - band_err[b] / (band_tgt[b] + 1e-8)
            for b in range(len(edges) - 1)}


@torch.no_grad()
def _divergence(field):
    """div = du_x/dx + du_y/dy + du_z/dz for field [B,3,X,Y,Z].
    x,z : periodic+uniform -> FFT derivative with physical dx,dz.
    y   : non-uniform      -> 2nd-order finite difference on the actual y_grid.
    """
    B, C, Xn, Yn, Zn = field.shape
    # --- x,z via FFT (physical wavenumber = 2*pi*fftfreq / spacing) ---
    F = torch.fft.fft2(field, dim=(2, 4), norm="ortho")          # over x(2), z(4)
    kx = torch.fft.fftfreq(Xn, device=field.device) * (2 * np.pi / dx)
    kz = torch.fft.fftfreq(Zn, device=field.device) * (2 * np.pi / dz)
    KX = kx.to(field.device)[None, None, :, None, None]
    KZ = kz.to(field.device)[None, None, None, None, :]
    div_xz = torch.fft.ifft2(1j * (KX * F[:, 0] + KZ * F[:, 2]), dim=(2, 4)).real
    # --- y via non-uniform finite difference (interior, 2nd order) ---
    # uy = field[:,1] is u_y [B,X,Y,Z]. Interior nodes j=1..Yn-2.
    #   h_m = y_j - y_{j-1}   (back spacing)   = y_grid[1:-1] - y_grid[:-2]
    #   h_p = y_{j+1} - y_j   (fwd  spacing)   = y_grid[2:]   - y_grid[1:-1]
    # 2nd-order weights:  f'[j] = -h_p/(h_m(h_m+h_p)) f[j-1]
    #                        + (h_p-h_m)/(h_m h_p)    f[j]
    #                        + h_m/(h_p(h_m+h_p))    f[j+1]
    uy = field[:, 1]                                   # [B,X,Y,Z]
    hm = torch.as_tensor(y_grid[1:-1] - y_grid[:-2], device=field.device)   # [Yn-2]
    hp = torch.as_tensor(y_grid[2:]   - y_grid[1:-1], device=field.device)  # [Yn-2]
    w0 = -hp / (hm * (hm + hp))                         # coeff of f[j-1]
    w1 = (hp - hm) / (hm * hp)                          # coeff of f[j]
    w2 = hm / (hp * (hm + hp))                          # coeff of f[j+1]
    du_y = torch.empty_like(uy)
    # uy is [B,X,Y,Z]; the 31 interior nodes are at dim 2 -> weight shape [1,1,31,1]
    du_y[:, :, 1:-1, :] = (w0[None, None, :, None] * uy[:, :, :-2, :]
                           + w1[None, None, :, None] * uy[:, :, 1:-1, :]
                           + w2[None, None, :, None] * uy[:, :, 2:, :])
    # boundary rows: 1st-order one-sided
    du_y[:, :, 0, :]  = (uy[:, :, 1, :] - uy[:, :, 0, :]) / dy[0]
    du_y[:, :, -1, :] = (uy[:, :, -1, :] - uy[:, :, -2, :]) / dy[-1]
    return div_xz + du_y


@torch.no_grad()
def val_divergence(model, ds):
    """Mean |div u_pred| and |div (u_pred - y)| in PHYSICAL space (physics sanity).
    Unnormalize first: per-component std scaling breaks incompressibility, so the
    divergence must be measured on the physical (unnormalized) field.
    For reference, div of the true DNS field here is ~5e-2 (discretization residual)."""
    model.eval()
    div_pred = div_err = div_ref = 0.0
    n = 0
    for x, y in ds:
        pred = model(x)
        pred_phys = unnormalize_velocity_gpu(pred)
        y_phys = unnormalize_velocity_gpu(y)
        div_pred += _divergence(pred_phys).abs().mean().item()
        div_err += _divergence(pred_phys - y_phys).abs().mean().item()
        div_ref += _divergence(y_phys).abs().mean().item()   # baseline: DNS residual
        n += 1
    return div_pred / n, div_err / n, div_ref / n


def train_loop(model, loss_fn, ds_train, ds_val, epochs, optim, lr, project_name, run_config=None):
    opti = optim(model.parameters(), lr=lr)
    cfg = {"epochs": epochs, "learning_rate": lr, "net_c": net_c, "net_n_blocks": net_n_blocks}
    if run_config:
        cfg.update(run_config)
    trackio.init(project=project_name, config=cfg)
    with torch.no_grad():
        base_skill = val_band_skill(model, ds_val)
        d0, de0, ref0 = val_divergence(model, ds_val)
    trackio.log({"epoch": -1,
                 **{f"val/skill/{n}": v for n, v in base_skill.items()},
                 "val/div_pred": d0, "val/div_err": de0, "val/div_ref(dns)": ref0})
    for epoch in range(epochs):
        tr = train(model, opti, loss_fn, ds_train)
        val_mse = evaluate(model, ds_val, loss_fn)
        skill = val_band_skill(model, ds_val)
        d, de, ref = val_divergence(model, ds_val)
        trackio.log({
            "epoch": epoch,
            "train/mse": tr,
            "val/mse": val_mse,
            "val/EV": 1.0 - val_mse,
            **{f"val/skill/{name}": v for name, v in skill.items()},
            "val/div_pred": d, "val/div_err": de, "val/div_ref(dns)": ref,
        })
    trackio.finish()


# In[ ]:


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
n_epochs = 20   # drop to 2 for a smoke test first
train_loop(
    model=model,
    loss_fn=mse_loss,
    ds_train=train_dl,
    ds_val=val_dl,
    epochs=n_epochs,
    optim=optim,
    lr=lr,
    project_name=project_name,
    run_config={"net_c": net_c, "net_n_blocks": net_n_blocks},
)


# In[ ]:




