# Full-channel reconstruction benchmark log

One experiment, one question. Keep the direct baseline fixed while changing one input, target, or training strategy at a time. Use this file as the experiment index; put plot and checkpoint links in the run log, not in the benchmark IDs. No scored runs under the new split yet.

## Data contract

- **Train:** A1, `s3://simulations/flow_export/A1/data.zarr`, `200:3196` for the current export. Training-only normalization; reuse the same statistics on validation.
- **Validation:** independent `s3://simulations/validation/re200-sr1etot/data.zarr`, `200:1001` for the current export. Use for development and checkpoint selection, not fitting normalization. Keep source indices `0–199` out of both runs to exclude the startup interval; this is a conservative cutoff, not a stationarity certificate.
- **Test:** original A1 tail `3607:3999`, not loaded by the trainer. Do not use it to choose configurations. Because it is from A1, report its results separately from independent-run validation.
- Record the actual slices for each run (a smoke test uses shorter selections). Do not compare scores across different target channels, grids, or validation slices as if they were the same task.
- `p_wall` means pressure at the chosen wall. `Q_DNS` means the exported full-volume `critq` on the selected target grid. `Q̂` is a model prediction available from pressure at inference. `U = (u, v, w)`.

## Benchmark queue

| ID | Configuration | Question and comparison | Status |
| --- | --- | --- | --- |
| B01 | `p_wall → Q_DNS` | Which wall distances and Fourier bands of Q are recoverable from wall pressure? Compare to a training mean-Q profile; keep this model as the Q predictor for B05. | Trainer task `wall_pressure_to_q`; not run on new split |
| B02 | `Q_DNS → U` | How much velocity can be reconstructed from the true volume Q alone? Diagnostic with privileged input, not a wall-only result or a formal upper bound. | Trainer task `full_q_to_velocity`; not run on new split |
| B03 | `p_wall → U` | Baseline: what can the wall-only model recover directly? Compare to a training mean-velocity profile; report one error profile per component. | Trainer task `wall_pressure_to_velocity`; not run on new split |
| B04 | `p_wall + Q_DNS → U` | Does access to true Q add predictive information over B03, and does wall pressure add information over B02? Oracle-input ablation only; match grid and capacity. | Planned |
| B05 | `p_wall → Q̂`, then `p_wall + Q̂ → U` | Does predicted Q improve over the deployable baseline B03? Compare also to B04 to see how much of the oracle gain survives. Start with a frozen B01 and use temporally blocked out-of-fold Q predictions to train stage two; never substitute true Q or fit either stage on validation. | Planned |
| B06 | `p_wall → ΔU`, with `Û = U_baseline + ΔÛ` | Does learning a correction improve over the frozen baseline or an equally sized direct model? Start with residuals relative to frozen B03; use out-of-fold baseline predictions for stage-two training. Report final velocity error and the full inference cost. | Planned |
| B07 | `p_wall → u` | Does a dedicated streamwise model beat the u channel of B03? Compare the same target, normalization, and evaluation points. | Planned |
| B08 | `p_wall → v` | Does a dedicated wall-normal model beat the v channel of B03? Check where in y and in which spectral bands any gain occurs. | Planned |
| B09 | `p_wall → w` | Does a dedicated spanwise model beat the w channel of B03? Compare the same target and include the cost of all three separate models versus one joint model. | Planned |

Suggested first pass: B03 as anchor, B01 and B02 to understand Q, then B07–B09 if the component errors justify them. Hold B04–B06 until the baseline plots and the training-only construction of second-stage inputs are in place. The trainer currently accepts one input channel; the stacked-input and residual cases are tracked here, not implemented by this change.

## Shared plots and checks

- **MSE vs wall distance (first plot):** unnormalize predictions and targets, then compute `mean_(snapshot,x,z)[(prediction - target)^2]` at every y for each component separately. Accumulate squared-error sums and sample counts, including partial batches. Plot against the actual grid distance to the instrumented wall, not the y index. Optionally add y+ with a documented reference friction velocity. Show the training mean-profile baseline on the same axes. Compare solo and joint models per component, not by three-channel average versus one-channel average.
- **Fourier:** distinguish target power, prediction power, and power of the prediction error. Matching target power does not show that the model recovered the right phases. Start with an x–z FFT at fixed y; check the normalization by comparing integrated error power to spatial MSE there. The current trainer bins errors by integer mode radius and reports normalized-field errors; label this explicitly and decide separately when to use dimensional wavenumbers or component-wise relative skill. Keep DC separate.
- **Q-criterion:** first confirm how the exporter computes and scales `critq` and which sign convention it uses. For B01 plot DNS versus predicted slices on a shared color scale, distributions and error by y, and error stratified by Q magnitude. If comparing Q derived from predicted velocity to predicted Q directly, document the derivatives, physical coordinate spacing, and boundary treatment; treat this as a separate diagnostic.
- **Energy:** keep fluid energy and compute consumption separate. For fluid analysis, specify whether the quantity is mean kinetic energy or fluctuation energy and whether you want pressure-gradient input and viscous dissipation. Write the budget and agree on units, volume weighting, and the needed force/gradient data before claiming budget closure. For compute, record wall time, device and memory first; report measured energy only when device telemetry is available, and include both stages or all three component models where applicable.
- **Comparability:** freeze wall, y grid, x/z stride, time slices, seed(s), training budget, and evaluation code for a comparison. Log per-component physical-unit errors and, if needed, training-normalized metrics. Do not use the test split to debug or tune. Existing trainer validation loss averages batches; use sample-weighted error for benchmark reporting so a short last batch is not overrepresented.

## Run log

| Run ID | Benchmark | Status | Data slices, seed, grid, model/settings | Results and artifact links | Decision / next step |
| --- | --- | --- | --- | --- | --- |
| — | — | Not run | — | — | Establish B03 and the MSE-vs-wall-distance plot. |

For each run, add the commit, normalization scope, best checkpoint epoch, parameter count, training and inference wall time, and links to the checkpoint and plots. Record smoke tests as such; they verify plumbing, not model skill. Keep failed runs with the reason rather than overwriting them.
