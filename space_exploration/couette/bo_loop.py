"""Bayesian-optimization driver for the Couette architecture search.

Uses `ax` (installed: ax-platform / botorch / gpytorch) with a **warm start**:
existing S3 checkpoints are evaluated once and fed to the surrogate as initial
observations, so the first BO trials start smarter than random.

Search space:
  lift_width  16..128   (log int)
  c3d         16..128   (log int)
  n_blocks    2..15     (int)
  widen       {0,32,64} (choice int)
  (spectral_lambda=0.2, epochs=20 fixed; epochs overridable via --epochs)

Objective: minimize best val MSE (train_run.run_trial).

Usage:
  # smoke test, no training-heavy archs, warm-start from existing checkpoints
  python bo_loop.py --n-trials 5 --epochs 2 --warm-start

  # real run
  python bo_loop.py --n-trials 20 --epochs 20 --warm-start --out bo_results.json

  # no warm start / random fallback (no ax needed)
  python bo_loop.py --random --n-trials 3 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from train_run import run_trial
import couette_common as cc
import eval_run as ev

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
LIFT_WIDTHS = [16, 32, 64, 128]
C3DS = [16, 32, 64, 128]
N_BLOCKS = [2, 3, 4, 5, 8, 10, 12, 15, 20]
WIDENS = [0, 1, 32, 64]


def sample_random() -> dict:
    import random
    return {
        "lift_width": random.choice(LIFT_WIDTHS),
        "c3d": random.choice(C3DS),
        "n_blocks": random.choice(N_BLOCKS),
        "widen": random.choice(WIDENS),
    }


def build_ax_search_space():
    """Build the ax SearchSpace over the arch knobs (integer params)."""
    from ax.core.search_space import SearchSpace
    from ax.core.parameter import RangeParameter, ChoiceParameter, ParameterType
    params = [
        RangeParameter(name="lift_width", parameter_type=ParameterType.INT,
                       lower=16, upper=128, log_scale=True),
        RangeParameter(name="c3d", parameter_type=ParameterType.INT,
                       lower=16, upper=128, log_scale=True),
        RangeParameter(name="n_blocks", parameter_type=ParameterType.INT,
                       lower=2, upper=20),
        ChoiceParameter(name="widen", parameter_type=ParameterType.INT,
                        values=[0, 1, 32, 64]),
    ]
    return SearchSpace(parameters=params)


def warm_start_points(existing: list[dict]) -> list[tuple[dict, float]]:
    """Convert a list of {arch..., val_mse} into (params, y) tuples for ax."""
    pts = []
    for e in existing:
        arch = e.get("arch", e)
        p = {k: int(arch[k]) for k in ("lift_width", "c3d", "n_blocks", "widen")}
        y = float(e["val_mse"])
        pts.append((p, y))
    return pts


def _benchmark(model, arch, device, log=True):
    """Run the full statistical evaluation on val, return a JSON-ready dict."""
    splits = cc.load_splits()
    mean, std = cc.normalization_stats(splits["all_data"])
    mean_gpu = torch.from_numpy(mean).to(device)
    std_gpu = torch.from_numpy(std).to(device)
    _, _, _, val_ds = cc.make_dataloaders(splits, mean, std, device)
    pred = cc.predict_phys(model, val_ds, device, mean_gpu, std_gpu)
    target = splits["val"][:, :3, :, :, :].astype("float32")
    res = ev.evaluate(pred, target, splits["y"])
    if log:
        print("  [bench] mse=%.4e ev=%.4f k_resolved=%.1f" %
              (res["mse_global"], res["ev_global"], res["k_resolved_est"]), flush=True)
    return res


def discover_existing() -> list[dict]:
    """Find all `best.pt` checkpoints under the bo_ prefix, load their config
    + val_mse (by evaluating each), and return [{arch, val_mse, s3_prefix}].

    This is the warm-start source: your 4 existing width-study checkpoints
    plus any bo_* runs already done.
    """
    import s3fs
    fs = s3fs.S3FileSystem(profile='default',
                           client_kwargs={'endpoint_url': cc.S3_ENDPOINT})
    base = f"{cc.S3_BUCKET}/{cc.S3_MODEL_PREFIX}"
    try:
        subdirs = [d.rstrip("/").split("/")[-1] for d in fs.ls(base)]
    except Exception:
        return []
    # only arch-named folders (bo_* and width-study_*)
    cands = [d for d in subdirs if d.startswith(("bo_", "width-study_"))]
    device = cc.get_device()
    splits = cc.load_splits()
    mean, std = cc.normalization_stats(splits["all_data"])
    mean_gpu = torch.from_numpy(mean).to(device)
    std_gpu = torch.from_numpy(std).to(device)
    _, _, _, val_ds = cc.make_dataloaders(splits, mean, std, device)
    target = splits["val"][:, :3, :, :, :].astype("float32")

    out = []
    for d in cands:
        prefix = f"{cc.S3_MODEL_PREFIX}/{d}"
        try:
            base_dir = f"{base}/{d}"
            files = [f.split("/")[-1] for f in fs.ls(base_dir)]
            # pick the best tag: prefer best.pt, else final.pt, else last epoch_NNN
            if "best.pt" in files:
                tag = "best"
            elif "final.pt" in files:
                tag = "final"
            else:
                eps = sorted([f for f in files if f.startswith("epoch_")])
                if not eps:
                    continue
                tag = eps[-1].replace(".pt", "")
            model, cfg = cc.load_model_from_s3(prefix, tag, device)
            pred = cc.predict_phys(model, val_ds, device, mean_gpu, std_gpu)
            val_mse = float(((pred - target) ** 2).mean())
            arch = {"lift_width": int(cfg["lift_width"]),
                    "c3d": int(cfg["c3d"]),
                    "n_blocks": int(cfg["n_blocks"]),
                    "widen": int(cfg.get("widen", 0))}
            out.append({"arch": arch, "val_mse": val_mse,
                        "s3_prefix": prefix, "tag": tag,
                        "n_params": cc.count_params(model)})
            print(f"  [warm] {d}/{tag}.pt: {arch}  val_mse={val_mse:.4f}", flush=True)
            del model
        except Exception as e:
            print(f"  [warm] skip {d}: {e}", flush=True)
    return out


def _write_progress(path: str, trials: list[dict], meta: dict):
    """Cheap per-trial progress file (best so far + each trial's val_mse)."""
    best = min(trials, key=lambda t: t["val_mse"]) if trials else None
    doc = {
        "updated": __import__("datetime").datetime.now().isoformat(),
        **meta,
        "n_done": len(trials),
        "best": ({"arch": best["arch"], "val_mse": best["val_mse"]}
                 if best else None),
        "trials": [{"arch": t["arch"], "val_mse": t["val_mse"],
                    "n_params": t["n_params"], "best_epoch": t["best_epoch"]}
                   for t in trials],
    }
    with open(path, "w") as f:
        json.dump(ev.to_jsonable(doc), f, indent=2)


def run_bo(n_trials: int, results_path: str, epochs: int | None = None,
           warm_start: list[dict] | None = None, seed: int = 0,
           trackio_project: str | None = "couette-bo"):
    """Run ax Bayesian optimization with optional warm-start observations."""
    import warnings
    warnings.filterwarnings("ignore")
    from ax import Client
    from ax.core.experiment import Experiment

    progress_path = results_path.replace(".json", "_progress.json")

    ss = build_ax_search_space()
    exp = Experiment(search_space=ss, name="couette-arch-bo")
    client = Client(random_seed=seed)
    client.set_experiment(exp)
    client.configure_optimization(objective="-val_mse")   # minimize
    assert exp.optimization_config.objective.minimize is True

    device = cc.get_device()

    # ---- warm start: feed existing (arch, val_mse) into the surrogate ----
    n_warm = 0
    if warm_start:
        for p, yval in warm_start_points(warm_start):
            # only warm-start points that lie inside the search space
            try:
                ss.validate_membership(parameters=p)
            except Exception as e:
                print(f"  [warm] skip {p} ({e})", flush=True)
                continue
            ti = client.attach_trial(parameters=p)
            client.attach_data(ti, {"val_mse": yval})
            client.complete_trial(ti)
            n_warm += 1
        print(f"warm-started {n_warm} existing checkpoints into the surrogate", flush=True)

    # ---- BO loop ----
    trials = []
    meta = {"n_trials": n_trials, "epochs": epochs, "n_warm_start": n_warm,
            "mode": "ax-bo"}
    _write_progress(progress_path, trials, meta)
    for i in range(n_trials):
        next_trials = client.get_next_trials(max_trials=1)
        trial_index = next(iter(next_trials))
        params = {k: int(v) for k, v in next_trials[trial_index].items()}
        print(f"\n===== BO trial {i} params={params} =====", flush=True)
        res = run_trial(params, epochs=epochs, log=True,
                        trackio_project=trackio_project,
                        trial_label=f"trial_{i}")
        bench = _benchmark(res["model"], res["arch"], device)
        del res["model"]
        res["benchmark"] = bench
        client.attach_data(trial_index, {"val_mse": res["val_mse"]})
        client.complete_trial(trial_index)
        trials.append(res)
        print(f"  -> val_mse={res['val_mse']:.4f}", flush=True)
        _write_progress(progress_path, trials, meta)

    best_params, best_data, _, _ = client.get_best_parameterization()
    print("\n===== BEST (ax) =====", flush=True)
    print("params:", best_params, " val_mse:", best_data)

    out = {"n_trials": n_trials, "epochs": epochs,
           "n_warm_start": n_warm,
           "trials": trials, "best": best_params}
    with open(results_path, "w") as f:
        json.dump(ev.to_jsonable(out), f, indent=2)
    print(f"results -> {results_path}")
    return trials, best_params


def run_random(n_trials: int, results_path: str, seed: int = 0,
               epochs: int | None = None,
               trackio_project: str | None = "couette-random"):
    import random
    random.seed(seed)
    device = cc.get_device()

    progress_path = results_path.replace(".json", "_progress.json")
    trials = []
    meta = {"n_trials": n_trials, "epochs": epochs, "mode": "random"}
    _write_progress(progress_path, trials, meta)
    for i in range(n_trials):
        p = sample_random()
        print(f"\n===== random trial {i} params={p} =====", flush=True)
        res = run_trial(p, epochs=epochs, log=True,
                        trackio_project=trackio_project,
                        trial_label=f"random_{i}")
        bench = _benchmark(res["model"], res["arch"], device)
        del res["model"]
        res["benchmark"] = bench
        trials.append(res)
        print(f"  -> val_mse={res['val_mse']:.4f}  ({res['n_params']/1e6:.2f}M)",
              flush=True)
        _write_progress(progress_path, trials, meta)

    best = min(trials, key=lambda t: t["val_mse"])
    print("\n===== BEST (random) =====")
    print(json.dumps({k: v for k, v in best.items() if k != "benchmark"}, indent=2))
    out = {"n_trials": n_trials, "epochs": epochs,
           "trials": trials, "best": best}
    with open(results_path, "w") as f:
        json.dump(ev.to_jsonable(out), f, indent=2)
    print(f"results -> {results_path}")
    return trials, best


if __name__ == "__main__":
    import torch  # ensure available for _benchmark
    ap = argparse.ArgumentParser(description="BO over Couette architecture")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training epochs (default 20); use 2 for smoke test")
    ap.add_argument("--warm-start", action="store_true",
                    help="seed the surrogate from existing S3 checkpoints")
    ap.add_argument("--random", action="store_true",
                    help="use random search instead of ax (no ax needed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bo_results.json")
    args = ap.parse_args()

    if args.random:
        run_random(args.n_trials, args.out, args.seed, epochs=args.epochs)
    else:
        warm = discover_existing() if args.warm_start else None
        if args.warm_start:
            print(f"discovered {len(warm)} existing checkpoints for warm-start")
        run_bo(args.n_trials, args.out, epochs=args.epochs,
               warm_start=warm, seed=args.seed)
