"""Train one Couette config and keep ONLY the best validation checkpoint.

Standalone (no notebook state). Usable two ways:

  1. As the BO objective:   from train_run import run_trial; run_trial(params)
  2. From the CLI:          python train_run.py --params '{"c3d":64,"n_blocks":10}'

The objective returned is the best val MSE (lower is better) — the scalar
that Bayesian optimization minimizes.
"""
from __future__ import annotations

import argparse
import json
import os
import torch

from couette_common import (
    get_device, load_splits, normalization_stats, make_dataloaders,
    make_model, SpectralLoss, save_model_to_s3, delete_s3,
    S3_MODEL_PREFIX,
)

# Fixed hyper-parameters (not part of the BO search)
EPOCHS = 20
LR = 5e-4
BATCH_SIZE = 4
OPTIM = torch.optim.Adam
SPECTRAL_LAMBDA = 0.2  # held fixed per user; not swept


def _val_mse(model, val_dl, device) -> float:
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            tot += torch.nn.functional.mse_loss(model(x), y).item()
    return tot / len(val_dl)


def _run_epoch(model, opti, loss_fn, train_dl):
    model.train()
    tot = 0.0
    for x, y in train_dl:
        opti.zero_grad()
        total, _, _ = loss_fn(model(x), y)
        total.backward()
        opti.step()
        tot += total.item()
    return tot / len(train_dl)


def _trackio_log(trial_label: str, epoch: int, train_err: float, val_mse: float,
                 arch: dict, best_so_far: float):
    """Log a training step to a Trackio dashboard (best-effort)."""
    try:
        import trackio
        trackio.log({
            "epoch": epoch,
            "train/loss": train_err,
            "val/mse": val_mse,
            "val/best_so_far": best_so_far,
            "arch/lift_width": arch["lift_width"],
            "arch/c3d": arch["c3d"],
            "arch/n_blocks": arch["n_blocks"],
            "arch/widen": arch["widen"],
            "trial": trial_label,
        })
    except Exception:
        pass  # tracking is optional; never let it break training


def run_trial(params: dict, *, device: str | None = None,
              log: bool = False, epochs: int | None = None,
              trackio_project: str | None = None,
              trial_label: str | None = None) -> dict:
    """Train one config; return dict with 'val_mse' (objective) + metadata.

    Only the BEST (lowest val-MSE) checkpoint is kept on S3; all per-epoch
    checkpoints are deleted as we go.
    """
    device = device or get_device()
    epochs = epochs or EPOCHS
    trial_label = trial_label or "trial"

    # params: architecture knobs (lift_width, c3d, n_blocks, widen)
    arch = {
        "lift_width": int(params.get("lift_width", 32)),
        "c3d": int(params.get("c3d", 32)),
        "n_blocks": int(params.get("n_blocks", 5)),
        "widen": int(params.get("widen", 0)),
    }

    if log:
        print(f"[trial] {arch}", flush=True)

    # optional trackio dashboard for this run
    _trackio_ok = False
    if trackio_project is not None:
        try:
            import trackio
            trackio.init(project=trackio_project,
                         config={"arch": arch, "epochs": epochs, "lr": LR})
            _trackio_ok = True
        except Exception as e:
            if log:
                print(f"  [warn] trackio init failed: {e}", flush=True)

    # data (cached in-process across trials)
    splits = load_splits()
    mean, std = normalization_stats(splits["all_data"])
    mean_gpu = torch.from_numpy(mean).to(device)
    std_gpu = torch.from_numpy(std).to(device)
    train_dl, val_dl, _, _ = make_dataloaders(splits, mean, std, device,
                                              batch_size=BATCH_SIZE)

    model = make_model(arch, device)
    xb, yb = next(iter(val_dl))
    loss_fn = SpectralLoss(yb, dim=(-2, -1), n_bands=6, kmin=1.0,
                           lambda_weight=SPECTRAL_LAMBDA).to(device)
    opti = OPTIM(model.parameters(), lr=LR)

    # S3 run folder (unique per arch)
    prefix = (f"{S3_MODEL_PREFIX}/bo_lw{arch['lift_width']}"
              f"_c{arch['c3d']}_b{arch['n_blocks']}_w{arch['widen']}")

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        train_err = _run_epoch(model, opti, loss_fn, train_dl)
        val_mse = _val_mse(model, val_dl, device)

        if log:
            print(f"  epoch {epoch:3d}  train={train_err:.4f}  val={val_mse:.4f}",
                  flush=True)
        if _trackio_ok:
            _trackio_log(trial_label, epoch, train_err, val_mse, arch, best_val)

        if val_mse < best_val:
            # keep only the best: save new, drop the previous best
            if best_epoch >= 0:
                delete_s3(prefix, f"best_{best_epoch:03d}")
            best_val = val_mse
            best_epoch = epoch
            save_model_to_s3(model, prefix, f"best_{epoch:03d}",
                             extra_cfg={"epochs": epochs, "lr": LR,
                                        "spectral_lambda": SPECTRAL_LAMBDA},
                             widen=arch["widen"])
            # also maintain a stable "best" alias
            save_model_to_s3(model, prefix, "best",
                             extra_cfg={"epochs": epochs, "lr": LR,
                                        "spectral_lambda": SPECTRAL_LAMBDA,
                                        "best_epoch": epoch},
                             widen=arch["widen"])
        else:
            # nothing new to keep; (no per-epoch checkpoint is saved)
            pass

    n_params = sum(p.numel() for p in model.parameters())
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
    import sys
    ap = argparse.ArgumentParser(description="Train one Couette config")
    ap.add_argument("--params", type=str, default=None,
                    help='JSON arch params, e.g. \'{"c3d":64,"n_blocks":10}\'')
    ap.add_argument("--epochs", type=int, default=None,
                    help="override EPOCHS (default %d); use 2 for a smoke test" % EPOCHS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    result = run_trial(_parse_params(args.params),
                       epochs=args.epochs,
                       log=not args.quiet)
    print(json.dumps(result, indent=2))
