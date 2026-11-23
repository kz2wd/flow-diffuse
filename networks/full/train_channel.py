"""Train one Channel-flow (A1) config and keep ONLY the best validation checkpoint.

Standalone (no notebook state). Mirrors the `full_2.ipynb` framework (DataNet,
dask-backed zarr S3 data, plain MSE loss) but runs as a script, and is inspired
by `space_exploration/couette/train_run.py`:

  1. As the BO objective:   from train_channel import run_trial; run_trial(params)
  2. From the CLI:          python train_channel.py --params '{"c":128,"n_blocks":6}'

The objective returned is the best val MSE (lower is better). Trackio runs also
log validation MSE contributions by XZ Fourier band and paired power spectra
for fixed validation samples at the midpoint of the selected y grid.

Differences from `full_2.ipynb`:
  - The whole dataset is assumed to fit in RAM (loaded once up front), so we
    build an in-memory dataset rather than re-opening zarr per worker.
  - An adaptive learning rate (ReduceLROnPlateau on val MSE) is used instead of
    a fixed LR.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import resource
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import zarr
import s3fs
from tqdm.auto import tqdm

if __package__:
    from .channel_metrics import SpectralValidation, render_spectral_slice
    from .channel_model import make_model
else:  # CLI and notebooks importing train_channel directly
    from channel_metrics import SpectralValidation, render_spectral_slice
    from channel_model import make_model

# ---------------------------------------------------------------------------
# S3 / dataset location (matches full_2.ipynb)
# ---------------------------------------------------------------------------
S3_BUCKET = "simulations"
STORE_PATH = "flow_export/A1"
S3_ENDPOINT = "http://localhost:8333"
# Save bucket: a dedicated `models` bucket, `channel-1` prefix
S3_MODEL_BUCKET = "models"
S3_MODEL_PREFIX = "models/channel-1"

DEFAULT_TASK = "full_q_to_velocity"
# (input channel, first target channel, number of target channels)
TASK_CHANNELS = {
    DEFAULT_TASK: (4, 0, 3),
    "wall_pressure_to_q": (3, 4, 1),
    "wall_pressure_to_velocity": (3, 0, 3),
}

# Fixed hyper-parameters (not part of the BO search)
EPOCHS = 20
BATCH_SIZE = 4
NUM_WORKERS = 16
SCHEDULER_PATIENCE = 3      # epochs without val improvement before dropping LR
SCHEDULER_FACTOR = 0.5      # LR multiplier on plateau
SCHEDULER_MIN_LR = 1e-6
SPECTRAL_IMAGE_SAMPLES = 2
SPECTRAL_IMAGE_EVERY = 5


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


def load_into_ram(z, idx, in_ch: int, out_ch: int,
                  chunk: int = 16, workers: int = 8,
                  progress: bool = True, *, out_start_ch: int = 0,
                  wall_y: int | None = None, half_y: bool = False,
                  xz_stride: int = 1):
    """Pull the selected timesteps into RAM once. Returns (x, y) float32 tensors.

    `idx` is a `range` (from compute_splits) or a slice.

    Loads in `chunk`-snapshot blocks with a thread pool (S3 reads are I/O-bound,
    so threads give a real speedup without dask's overhead). Progress is reported
    via tqdm, and peak RSS is printed at the end so you can see the RAM cost.

    Each chunk is indexed by a **slice** (zarr only accepts int/slice), built from
    the chunk's source start/stop/step so strided splits stay correct.

    `half_y` keeps the lower half including the center plane on odd grids.
    Wall inputs still use the specified wall. `xz_stride` subsamples x and z
    on both inputs and targets without averaging.
    """
    nx, ny, nz = z.shape[2:]
    if xz_stride < 1:
        raise ValueError("xz_stride must be positive")
    if wall_y not in (None, 0, ny - 1):
        raise ValueError(f"wall_y must select the first or last y plane (0 or {ny - 1})")
    target_ny = (ny + 1) // 2 if half_y else ny
    target_y = slice(0, target_ny)
    input_y = target_y if wall_y is None else slice(wall_y, wall_y + 1)
    xz = slice(None, None, xz_stride)
    nx_out, nz_out = len(range(0, nx, xz_stride)), len(range(0, nz, xz_stride))
    target_channels = slice(out_start_ch, out_start_ch + out_ch)
    if isinstance(idx, range):
        r = idx
    elif isinstance(idx, slice):
        r = range(*idx.indices(z.shape[0]))
    else:
        r = range(*list(idx))

    n = len(r)  # number of snapshots selected
    step = r.step if r.step and r.step > 0 else 1

    # pre-allocate output buffers (avoids a second big allocation / copy)
    shape_x = (n, 1, nx_out, target_ny if wall_y is None else 1, nz_out)
    shape_y = (n, out_ch, nx_out, target_ny, nz_out)
    x_out = np.empty(shape_x, dtype=np.float32)
    y_out = np.empty(shape_y, dtype=np.float32)

    # chunk boundaries in *selected-snapshot* coordinates (0..n)
    bounds = list(range(0, n, chunk))

    def _load_chunk(out_start: int):
        # map the chunk's selected-snapshot range back to source indices
        out_end = min(out_start + chunk, n)
        # selected index k in this chunk corresponds to source index:
        #   src = r.start + (out_start + k) * step
        src_first = r.start + out_start * step
        src_last = r.start + (out_end - 1) * step
        s = slice(src_first, src_last + 1, step)   # zarr accepts a slice
        import s3fs as _s3fs, zarr as _zarr
        fs = _s3fs.S3FileSystem(profile="default",
                                client_kwargs={"endpoint_url": S3_ENDPOINT})
        za = _zarr.open_array(
            store=_s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/data.zarr", s3=fs),
            mode="r")
        xc = np.asarray(za[s, in_ch:in_ch + 1, xz, input_y, xz], dtype=np.float32)
        yc = np.asarray(za[s, target_channels, xz, target_y, xz], dtype=np.float32)
        return out_start, xc, yc

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_load_chunk, s) for s in bounds]
        completed = as_completed(futs)
        if progress:
            completed = tqdm(completed, total=len(futs), desc="Load Zarr",
                             unit="chunk", file=sys.stdout)
        for fut in completed:
            out_start, xc, yc = fut.result()
            out_end = min(out_start + chunk, n)
            assert xc.shape[0] == out_end - out_start, \
                f"chunk yielded {xc.shape[0]} rows, expected {out_end - out_start}"
            x_out[out_start:out_end] = xc
            y_out[out_start:out_end] = yc

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if progress:
        print(f"  [load] {n} snapshots  "
              f"x={x_out.nbytes / 1024**3:.2f} GiB  "
              f"y={y_out.nbytes / 1024**3:.2f} GiB  "
              f"peak RSS={rss_after / 1024**2:.2f} GiB", flush=True)

    return torch.from_numpy(x_out), torch.from_numpy(y_out)


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
                     batch_size: int, num_workers: int, device: str, *,
                     shuffle: bool = False):
    """Build a normalized-batches generator + dataloader.

    Normalization is applied on-device (in place) to avoid extra CPU copies,
    mirroring the `normalized_batches` helper in full_2.ipynb.
    """
    ds = RAMDataset(x, y)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
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
def _val_mse(model, normalized_batches, val_dl, loss_fn,
             spectral: SpectralValidation | None = None) -> float:
    model.eval()
    tot = 0.0
    with torch.no_grad():
        pbar = tqdm(normalized_batches(val_dl), total=len(val_dl),
                    desc="Validation", unit="batch", file=sys.stdout)
        for batch_num, (x, y) in enumerate(pbar, start=1):
            prediction = model(x)
            tot += loss_fn(prediction, y).item()
            if spectral is not None:
                spectral.update(prediction, y)
            del prediction
            pbar.set_postfix(loss=f"{tot / batch_num:.4f}", refresh=False)
    return tot / len(val_dl)


def _run_epoch(model, opti, loss_fn, normalized_batches, train_dl):
    model.train()
    tot = 0.0
    pbar = tqdm(normalized_batches(train_dl), total=len(train_dl),
                desc="Train", unit="batch", file=sys.stdout)
    for batch_num, (x, y) in enumerate(pbar, start=1):
        opti.zero_grad()
        l = loss_fn(model(x), y)
        l.backward()
        opti.step()
        tot += l.item()
        pbar.set_postfix(loss=f"{tot / batch_num:.4f}", refresh=False)
    return tot / len(train_dl)


def _trackio_log(trial_label: str, epoch: int, train_err: float, val_mse: float,
                 arch: dict, best_so_far: float, lr_now: float,
                 spectral: SpectralValidation, channel_names: tuple[str, ...],
                 *, verbose: bool = False):
    """Log a training step to a Trackio dashboard (best-effort)."""
    try:
        import trackio
        metrics = {
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
            **spectral.scalars(),
        }
        try:
            for sample, power in enumerate(spectral.slices):
                metrics[f"val/spectral_xz/sample_{sample}"] = trackio.Image(
                    render_spectral_slice(power, channel_names),
                    caption=f"Validation sample {sample}, y index {spectral.y_index} "
                            "(selected grid), normalized fields",
                )
        except Exception as e:
            if verbose:
                print(f"  [warn] spectral image creation failed: {e}", flush=True)
        trackio.log(metrics)
    except Exception as e:
        if verbose:
            print(f"  [warn] trackio log failed: {e}", flush=True)


@dataclass(frozen=True)
class TrainingData:
    """CPU tensors and train-only normalization stats shared across trials."""

    train: tuple[torch.Tensor, torch.Tensor]
    val: tuple[torch.Tensor, torch.Tensor]
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    task: str = DEFAULT_TASK
    wall: str | None = None
    half_y: bool = False
    xz_stride: int = 1


def load_training_data(*, max_samples: int | None = None, log: bool = False,
                       task: str = DEFAULT_TASK, wall: str | None = None,
                       half_y: bool = False, xz_stride: int = 1) -> TrainingData:
    """Load the selected task and spatial grid into RAM."""
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    if task not in TASK_CHANNELS:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(TASK_CHANNELS)}")
    if task == DEFAULT_TASK:
        if wall is not None:
            raise ValueError("wall applies only to wall pressure tasks")
    elif wall is None:
        wall = "lower"
    elif wall not in ("lower", "upper"):
        raise ValueError("wall must be 'lower' or 'upper'")

    z = _open_data_zarr()
    T = z.shape[0]
    in_ch, out_start_ch, out_ch = TASK_CHANNELS[task]
    wall_y = None if wall is None else 0 if wall == "lower" else z.shape[3] - 1
    if z.shape[1] < max(in_ch + 1, out_start_ch + out_ch):
        raise ValueError(f"Zarr array has too few channels for task {task!r}")
    splits = compute_splits(T)
    train_range = splits["train"][:max_samples]
    val_range = splits["val"][:max_samples]
    if not train_range or not val_range:
        raise ValueError(f"Expected nonempty train and validation splits for T={T}")
    if log:
        print(f"  task={task}  wall={wall}  half_y={half_y}  xz_stride={xz_stride}  "
              f"T={T}  max_samples={max_samples}", flush=True)
        for name, selected in (("train", train_range), ("val", val_range), ("test", splits["test"])):
            status = "held out, not loaded" if name == "test" else "loading"
            print(f"  {name}: [{selected.start}:{selected.stop}:{selected.step}] "
                  f"({len(selected)}/{len(splits[name])} snapshots; {status})", flush=True)

    selection = dict(in_ch=in_ch, out_ch=out_ch, out_start_ch=out_start_ch,
                     wall_y=wall_y, half_y=half_y, xz_stride=xz_stride)
    xtr, ytr = load_into_ram(z, train_range, **selection)
    xva, yva = load_into_ram(z, val_range, **selection)
    x_mean, x_std = _channel_stats(xtr)
    y_mean, y_std = _channel_stats(ytr)
    return TrainingData(train=(xtr, ytr), val=(xva, yva),
                        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                        task=task, wall=wall, half_y=half_y, xz_stride=xz_stride)


def run_trial(params: dict, *, data: TrainingData | None = None,
              device: str | None = None,
              log: bool = False, epochs: int | None = None,
              lr: float | None = None,
              max_samples: int | None = None,
              task: str | None = None, wall: str | None = None,
              half_y: bool | None = None, xz_stride: int | None = None,
              trackio_project: str | None = None,
              trial_label: str | None = None,
              spectral_image_samples: int = SPECTRAL_IMAGE_SAMPLES,
              spectral_image_every: int = SPECTRAL_IMAGE_EVERY) -> dict:
    """Train one config; return dict with 'val_mse' (objective) + metadata.

    Keep the best validation checkpoint under the Trackio project/run path
    (or a unique untracked path). On a keyboard interrupt during the epoch
    loop, also save the current weights as a separate checkpoint.
    Adaptive LR: ReduceLROnPlateau steps on val MSE.

    Pass `data=load_training_data(...)` to reuse RAM data and statistics across
    trials. Without `data`, this function loads train and validation itself;
    `max_samples`, `task`, `wall`, `half_y`, and `xz_stride` control loading in that case.
    Set `spectral_image_samples=0` to disable Trackio images. Otherwise, log them
    on the first epoch, every `spectral_image_every` epochs, and the final epoch.
    """
    if spectral_image_samples < 0 or spectral_image_every < 1:
        raise ValueError("spectral_image_samples must be nonnegative and spectral_image_every positive")
    if data is None:
        data = load_training_data(max_samples=max_samples, log=log,
                                  task=DEFAULT_TASK if task is None else task, wall=wall,
                                  half_y=bool(half_y), xz_stride=1 if xz_stride is None else xz_stride)
    elif any(value is not None for value in (max_samples, task, wall, half_y, xz_stride)):
        raise ValueError("Set loading options in load_training_data when using prepared data")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or EPOCHS
    lr = lr or 1e-3
    trial_label = trial_label or "trial"

    target = data.train[1]
    arch = {
        "c": int(params.get("c", 64)),
        "n_blocks": int(params.get("n_blocks", 3)),
        "dy": int(params.get("dy", target.shape[3])),
        "out_ch": target.shape[1],
    }
    if arch["dy"] != target.shape[3]:
        raise ValueError(f"dy={arch['dy']} does not match target depth {target.shape[3]}")
    run_config = {"arch": arch, "task": data.task, "wall": data.wall,
                  "half_y": data.half_y, "xz_stride": data.xz_stride,
                  "epochs": epochs, "lr": lr}

    if log:
        print(f"[trial] {run_config}", flush=True)

    trackio_run = None
    if trackio_project is not None:
        try:
            import trackio
            trackio_run = trackio.init(
                project=trackio_project,
                embed=False,
                config={**run_config,
                        "scheduler": "ReduceLROnPlateau",
                        "patience": SCHEDULER_PATIENCE,
                        "factor": SCHEDULER_FACTOR,
                        "spectral_image_samples": spectral_image_samples,
                        "spectral_image_every": spectral_image_every},
            )
        except Exception as e:
            if log:
                print(f"  [warn] trackio init failed: {e}", flush=True)

    train_dl, train_batches = make_dataloaders(
        *data.train, data.x_mean, data.y_mean, data.x_std, data.y_std,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device, shuffle=True)
    val_dl, val_batches = make_dataloaders(
        *data.val, data.x_mean, data.y_mean, data.x_std, data.y_std,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device)

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

    if trackio_run is not None:
        prefix = f"{S3_MODEL_PREFIX}/{trackio_run.project}/{trackio_run.name}"
    else:
        prefix = f"{S3_MODEL_PREFIX}/untracked/{uuid4().hex}"
    best_val = float("inf")
    best_epoch = -1
    interrupted_checkpoint_uri = None
    epoch = -1
    channel_names = ("Q",) if data.task == "wall_pressure_to_q" else ("u", "v", "w")
    try:
        for epoch in range(epochs):
            train_err = _run_epoch(model, opti, loss_fn, train_batches, train_dl)
            spectral = None
            if trackio_run is not None:
                log_images = (epoch == 0 or (epoch + 1) % spectral_image_every == 0
                              or epoch == epochs - 1)
                spectral = SpectralValidation(image_samples=spectral_image_samples if log_images else 0)
            val_mse = _val_mse(model, val_batches, val_dl, loss_fn, spectral)
            scheduler.step(val_mse)
            lr_now = opti.param_groups[0]["lr"]

            if log:
                print(f"  epoch {epoch:3d}  train={train_err:.4f}  val={val_mse:.4f}"
                      f"  lr={lr_now:.2e}", flush=True)
            if spectral is not None:
                _trackio_log(trial_label, epoch, train_err, val_mse, arch,
                             min(best_val, val_mse), lr_now, spectral, channel_names,
                             verbose=log)

            if val_mse < best_val:
                save_model_to_s3(model, prefix, f"best_{epoch:03d}",
                                 extra_cfg=run_config)
                save_model_to_s3(model, prefix, "best",
                                 extra_cfg={**run_config, "best_epoch": epoch})
                previous_best = best_epoch
                best_val, best_epoch = val_mse, epoch
                if previous_best >= 0:
                    delete_s3(prefix, f"best_{previous_best:03d}")
    except KeyboardInterrupt:
        print("  Training interrupted; saving current model weights...", flush=True)
        save_model_to_s3(model, prefix, "interrupted",
                         extra_cfg={**run_config, "interrupted_epoch": epoch})
        interrupted_checkpoint_uri = f"s3://{_checkpoint_key(prefix, 'interrupted')}"
        print(f"  Saved {interrupted_checkpoint_uri}", flush=True)
    finally:
        if trackio_run is not None:
            try:
                import trackio
                trackio.finish()
            except Exception:
                pass

    return {
        "val_mse": best_val,
        "best_epoch": best_epoch,
        "n_params": n_params,
        **run_config,
        "s3_prefix": prefix,
        "s3_best_tag": "best" if best_epoch >= 0 else None,
        "checkpoint_uri": (f"s3://{_checkpoint_key(prefix, 'best')}"
                           if best_epoch >= 0 else interrupted_checkpoint_uri),
        "interrupted_checkpoint_uri": interrupted_checkpoint_uri,
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
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap samples loaded per split (fast arch tests)")
    ap.add_argument("--task", choices=tuple(TASK_CHANNELS), default=DEFAULT_TASK)
    ap.add_argument("--wall", choices=("lower", "upper"),
                    help="pressure wall; defaults to lower on wall pressure tasks")
    ap.add_argument("--half-y", action="store_true", help="keep lower half of the y grid")
    ap.add_argument("--xz-stride", type=int, default=1,
                    help="take every Nth grid point in x and z")
    ap.add_argument("--project", type=str, default=None,
                    help="trackio project name (default: none)")
    ap.add_argument("--label", type=str, default="trial",
                    help="trial label for trackio")
    ap.add_argument("--spectral-image-samples", type=int, default=SPECTRAL_IMAGE_SAMPLES,
                    help="fixed validation samples per Trackio image step (0 disables images)")
    ap.add_argument("--spectral-image-every", type=int, default=SPECTRAL_IMAGE_EVERY,
                    help="Trackio image interval in epochs (also logs first and last epochs)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    result = run_trial(
        _parse_params(args.params),
        epochs=args.epochs,
        lr=args.lr,
        max_samples=args.max_samples,
        task=args.task,
        wall=args.wall,
        half_y=args.half_y,
        xz_stride=args.xz_stride,
        trackio_project=args.project,
        trial_label=args.label,
        spectral_image_samples=args.spectral_image_samples,
        spectral_image_every=args.spectral_image_every,
        log=not args.quiet,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "model"},
                     indent=2))
