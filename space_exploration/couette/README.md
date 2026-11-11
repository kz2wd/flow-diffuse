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

| Knob | Values |
|------|--------|
| `lift_width` | 16 / 32 / 64 / 128 |
| `c3d` | 16 / 32 / 64 / 128 |
| `n_blocks` | 2 / 3 / 4 / 5 / 8 / 10 / 12 / 15 |
| `widen` | 0 / 32 / 64 |

Fixed: `spectral_lambda=0.2`, `epochs=20`, `lr=5e-4`, `batch_size=4`.

## Quick smoke test (no ax required)

```bash
cd space_exploration/couette
python train_run.py --params '{"c3d":32,"n_blocks":2,"lift_width":16,"widen":0}'
python bo_loop.py --random --n-trials 3
```

## Full Bayesian optimization

```bash
uv pip install ax-platform   # pulls botorch + gpytorch
python bo_loop.py --n-trials 20 --out bo_results.json
```

## Saving / loading a benchmark (intermediate data)

```python
from couette_common import load_model_from_s3, predict_phys, get_device, \
    load_splits, normalization_stats, make_dataloaders, S3_MODEL_PREFIX
from eval_run import evaluate, save_results

# train:  python train_run.py --params '{...}'   -> best ckpt at {prefix}/best.pt
```

## Note on checkpointing

`train_run.run_trial` saves a checkpoint every epoch to a per-arch S3 folder
(`bo_lw.._c.._b.._w..`) and keeps **only the best** (by val MSE) — both as
`best_<epoch>.pt` and a stable `best.pt` alias. All other per-epoch
checkpoints are deleted as a better one is found.
