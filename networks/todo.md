# TODO — wall pressure → full velocity operator

Research document: building a **data + physics** based operator from wall pressure
to the full 3D velocity field. Diffusion-like in structure (iterative refinement,
network predicts a *correction*), but physics-driven (corrections target NS
residuals, not a data-density score).

## Case 1

- **Data-based initial estimation** — `u₀ = DataNet(p_wall)` (conv3d from the start)
- **Possible pre-correction** — Hodge-project `u₀` (make div-free, FFT, free) so the
  refinement loop starts on the divergence-free manifold
- **Correction steps** (repeat K times; `R(u)` = physics residual, e.g. `∇·u`, momentum, wall BC)
  - **1.1** `Δu = Net(u, R(u))` — correction sees field + residual (most data-driven)
  - **1.2** `Δu = Net(R(u))` — correction sees residual only (clean data/physics split) ← start here
  - **1.3** `Δu = L(R(u))` — learned *linear/FFT-diagonal* operator on residual (most physics, baseline)

Dial: 1.3 (most physics) ← 1.2 (balanced) ← 1.1 (most data). Compare all to answer
"how much data-driven?" with evidence.


Note:
5% reference divergence error on reference data: locate it, may be near wall, then what to think about it? is it okay?
- build analysis of predictions first
