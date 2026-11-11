"""Bayesian-optimization driver for the Couette architecture search.

Uses `ax` (https://github.com/ax-lib/ax) if available; otherwise falls back to
a simple random search so you can smoke-test the objective without installing
ax/botorch.

Search space (per user):
  lift_width  in {16, 32, 64, 128}   (log)
  c3d         in {16, 32, 64, 128}   (log)
  n_blocks    in {2,3,4,5,8,10,12,15}
  widen       in {0, 32, 64}          (0 = off)
  (spectral_lambda held at 0.2, epochs at 20 -- fixed)

Objective: minimize best val MSE (train_run.run_trial).

Usage:
  # install first:  uv pip install ax-platform   (brings botorch/gpytorch)
  python bo_loop.py --n-trials 20
  python bo_loop.py --random --n-trials 8        # smoke test, no ax needed
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from train_run import run_trial

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
LIFT_WIDTHS = [16, 32, 64, 128]
C3DS = [16, 32, 64, 128]
N_BLOCKS = [2, 3, 4, 5, 8, 10, 12, 15]
WIDENS = [0, 32, 64]


def sample_random() -> dict:
    import random
    return {
        "lift_width": random.choice(LIFT_WIDTHS),
        "c3d": random.choice(C3DS),
        "n_blocks": random.choice(N_BLOCKS),
        "widen": random.choice(WIDENS),
    }


def build_ax_space():
    """Build an ax SearchSpace over the arch knobs. Lazy import."""
    import ax
    from ax.service.utils import get_sobol

    search_space = ax.SearchSpace(
        parameters=[
            ax.range_param(name="lift_width", type="int", lower=16, upper=128),
            ax.range_param(name="c3d", type="int", lower=16, upper=128),
            ax.range_param(name="n_blocks", type="int", lower=2, upper=15),
            ax.range_param(name="widen", type="int", lower=0, upper=64),
        ],
        parameter_constrains=None,
        outcome_names=["val_mse"],
        minimize=True,
    )
    return search_space


def objective(params: dict) -> float:
    """AX objective: minimize val_mse. ax passes a params dict of our ranges."""
    # ax may pass numpy ints; coerce
    p = {k: int(v) for k, v in params.items()}
    res = run_trial(p, log=True)
    return res["val_mse"]


def run_bo(n_trials: int, results_path: str):
    import ax
    from ax.service.ax_client import AxClient
    from ax.service.metrics import Metric

    client = AxClient(search_space=build_ax_space())

    def metric(val_mse):
        return {"val_mse": val_mse}

    trials = []
    for i in range(n_trials):
        trial_index = client.get_next_experiment(
            model_arg="sobol" if i < 5 else None,
        )
        arm = client.get_arm(trial_index)
        params = arm.parameters
        print(f"\n===== trial {i} params={params} =====", flush=True)
        val_mse = objective(params)
        client.log_trial_data(
            trial_index=trial_index,
            data={"val_mse": {"mean": val_mse, "sem": 0.0}},
        )
        trials.append({"params": {k: int(v) for k, v in params.items()},
                       "val_mse": val_mse, "trial_index": trial_index})
        print(f"  -> val_mse={val_mse:.4f}", flush=True)

    best = client.get_best_parameters()
    print("\n===== BEST =====", flush=True)
    print(json.dumps(best, indent=2))
    with open(results_path, "w") as f:
        json.dump({"trials": trials, "best": best}, f, indent=2)
    print(f"results -> {results_path}")
    return trials, best


def run_random(n_trials: int, results_path: str, seed: int = 0):
    import random
    random.seed(seed)
    trials = []
    for i in range(n_trials):
        p = sample_random()
        print(f"\n===== random trial {i} params={p} =====", flush=True)
        res = run_trial(p, log=True)
        trials.append({"params": p, "val_mse": res["val_mse"],
                       "n_params": res["n_params"]})
        print(f"  -> val_mse={res['val_mse']:.4f}  ({res['n_params']/1e6:.2f}M)",
              flush=True)
    best = min(trials, key=lambda t: t["val_mse"])
    print("\n===== BEST (random) =====")
    print(json.dumps(best, indent=2))
    with open(results_path, "w") as f:
        json.dump({"trials": trials, "best": best}, f, indent=2)
    print(f"results -> {results_path}")
    return trials, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BO over Couette architecture")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--random", action="store_true",
                    help="use random search instead of ax (no ax needed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="bo_results.json")
    args = ap.parse_args()

    if args.random:
        run_random(args.n_trials, args.out, args.seed)
    else:
        try:
            run_bo(args.n_trials, args.out)
        except ImportError:
            print("ax not installed; falling back to random search. "
                  "Install with:  uv pip install ax-platform")
            run_random(args.n_trials, args.out, args.seed)
