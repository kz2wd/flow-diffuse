# space_exploration / couette

Clean, script-based pipeline for the Couette pressure→velocity architecture
search. The existing notebooks (`networks/framework_2.ipynb`,
`networks/couette_predi/*`) are left untouched; these modules are the new
single source of truth for training + evaluation + Bayesian optimization.

## Files

| File | Role |
|------|------|
| `couette_common.py` | data loading, `M4`, `SpectralLoss`, S3 save/load (shared) |
| `train_run.py` | train one config; **keep only the best** checkpoint; returns val MSE |
| `eval_run.py` | all statistical metrics → flat dict → JSON (intermediate data) |
| `bo_loop.py` | `ax` Bayesian-optimization driver (random-search fallback) |

## Search space

| Knob | Range |
|------|-------|
| `lift_width` | 16–128 (log, int) |
| `c3d` | 16–128 (log, int) |
| `n_blocks` | 2–20 (int) |
| `widen` | {0, 1, 32, 64} |

Fixed: `spectral_lambda=0.2`, `lr=5e-4`, `batch_size=4`. `--epochs` defaults to 20.

> `widen=1` is a real knob (the original `w1` models are thin but competitive),
> so it's included in the choice set.

## Quick smoke test (no ax required, 2 epochs)

```bash
cd space_exploration/couette
python train_run.py --params '{"c3d":32,"n_blocks":2,"lift_width":16,"widen":0}' --epochs 2
python bo_loop.py --random --n-trials 3 --epochs 2
```

## Bayesian optimization (ax, warm-started)

`ax-platform` is installed. `--warm-start` seeds the surrogate with every
existing S3 checkpoint (your `width-study_*` models + any `bo_*` runs) so the
first trials start near the known-good region instead of randomly.

```bash
# smoke test (2 epochs, 2 trials)
python bo_loop.py --n-trials 2 --epochs 2 --warm-start

# real run
python bo_loop.py --n-trials 20 --epochs 20 --warm-start --out bo_results.json
```

Warm-start points outside the search space are skipped (logged); widen the
space if you see skips for a knob you actually use.

## Monitoring a long run

Launch it in the background (or in tmux) and watch three ways:

```bash
cd space_exploration/couette

# 1) live log (full stream: warm-start, each trial, per-epoch train/val)
python bo_loop.py --n-trials 20 --epochs 20 --warm-start --out bo_results.json \
    2>&1 | tee bo_run.log
# ...then in another terminal:
tail -f bo_run.log

# 2) progress file (best-so-far + per-trial val_mse, updated after every trial)
watch -n 5 cat bo_results_progress.json
# or:  python -c "import json;d=json.load(open('bo_results_progress.json'));print(d['best'])"

# 3) Trackio dashboard (live per-epoch train/val curves per trial)
#    the run prints a URL; or:
trackio show --project "couette-bo"
```

- **`bo_run.log`** — everything (tee'd stdout). `grep "BO trial" bo_run.log` to jump
  between trials; `grep "val=" bo_run.log | tail` to watch convergence.
- **`bo_results_progress.json`** — the cheap view: `n_done`, `best`, and each
  trial's `{arch, val_mse}`. Safe to poll while the run is in progress.
- **Trackio** — `couette-bo` project, one run per trial, each with `train/loss`,
  `val/mse`, `val/best_so_far` per epoch. The warm-start incumbent is visible as
  the floor the trials try to beat.

The final `bo_results.json` (full benchmarks, written at the end) is what the
web page will read.

Per trial (in `bo_results.json`):
- `arch` — the architecture (`lift_width, c3d, n_blocks, widen`)
- `val_mse` — the BO objective (best over epochs)
- `n_params`, `best_epoch`, `s3_prefix` — where the weights live
- **`benchmark`** — the full `eval_run.evaluate()` result (26 keys):
  `mse_global`, `ev_global`, `mse_per_comp`, `mean/std_pred/tgt` profiles,
  `rms_pred/tgt`, `reynolds_*`, `Ek_pred/tgt` (per |k| annulus),
  `rel_spectral_err`, `k_resolved_est`, `frac_energy_above_kresolved`,
  `skew/kurtosis_center`. This is the intermediate data the comparison
  web page will read — no retraining needed to re-rank or plot.

On S3: the best *weights* per arch (`{prefix}/best.pt` + `best_<epoch>.pt`),
keeping only the best epoch.

## Note on checkpointing

`train_run.run_trial` saves a checkpoint every epoch to a per-arch S3 folder
(`bo_lw.._c.._b.._w..`) and keeps **only the best** (by val MSE) — both as
`best_<epoch>.pt` and a stable `best.pt` alias. All other per-epoch
checkpoints are deleted as a better one is found.
