# DNS Validation Plan — Reτ=200 Channel Flow (incompact3d)

**Case:** turbulent channel flow, xz-periodic
**Grid:** 128 × 129 × 128 (x × y × z)
**Domain:** Lx = 12.57 (≈4πδ), Ly = 2.0 (δ=1, wall-normal), Lz = 4.19 (≈4/3 πδ)
**Target:** Re_τ = 200

This document lists the validation metrics to implement, grouped by category, with target values / tolerances and implementation notes. Metrics are ordered roughly by implementation cost vs diagnostic power — cheapest and most diagnostic first.

---

## 1. Solver-correctness checks (residuals & BCs)

These don't need any reference paper — they check internal consistency of the simulation itself. Do these first; if they fail, nothing downstream is trustworthy.

| Metric | Target | Tolerance | Notes |
|---|---|---|---|
| Global mass conservation, ∫∇·u dV | 0 | < 1e-6 (or solver's native divergence tolerance) | Should already be enforced by the pressure projection; a violation points to a bug in postprocessing or an unconverged Poisson solve, not physics. |
| No-slip BC, u,v,w at y=0 and y=2δ | 0 | < 1e-10 (machine/BC-enforcement precision) | Check both walls separately — asymmetry between walls is a red flag. |
| Periodicity in x, z | first plane = last plane | exact (< 1e-12) | Catches indexing/implementation bugs rather than physical error. |
| Mean momentum balance closure | τ_total(y) = ν d⟨u⟩/dy − ⟨u'v'⟩ linear in y | slope error < 2–3% | **High value check.** Plot viscous stress, Reynolds stress, and their sum vs y; sum must be linear from τ_w at wall to 0 at centerline. Very sensitive to statistics not being converged or to subtle bugs. |
| Statistical stationarity | d⟨u⟩/dt ≈ 0 over averaging window | drift < 1% of u_τ over the averaging window | Confirms the flow has left the transient and averaging window is valid. |

---

## 2. Reynolds number checks

| Metric | Target | Tolerance | Notes |
|---|---|---|---|
| Re_τ from wall shear: u_τ = sqrt(ν·du/dy\|_wall), Re_τ = u_τ·δ/ν | 200 | ± 2–3% (± 5% acceptable at this resolution) | Average both walls; report both individually first to check for asymmetry (should agree to <1%). |
| Re_b (bulk, from mean streamwise velocity integrated over y) | ≈ 3300 (Reτ=200 reference channels, e.g. Moser/Kim/Mansour 1999) | ± 5% | Depends on forcing type (CFR vs CPG) — if incompact3d uses constant flow rate forcing, Re_b is a free/exact check; if constant pressure gradient, Re_τ is the free one and Re_b is the derived check. Confirm which forcing incompact3d uses for this case. |
| Dean-correlation cross-check: Re_τ ≈ 0.09·Re_b^0.88 | consistent with above | ± 10% (correlation itself is approximate) | Independent sanity check, not a primary metric. |

---

## 3. Mean flow statistics (compare to universal law-of-the-wall constants — no paper table needed)

| Metric | Target | Tolerance | Notes |
|---|---|---|---|
| Viscous sublayer: u+ = y+ | exact for y+ < 5 | deviation < 5% for y+ < 5 | |
| Log-law region: u+ = (1/κ)ln(y+) + B, κ≈0.41, B≈5.2 | slope/intercept match | κ within ± 0.02, B within ± 0.3 | At Reτ=200 the log-law region is short (roughly 30 < y+ < 100) — don't expect a long clean fit; judge qualitatively as well as quantitatively. |

---

## 4. Turbulence statistics (needs a small number of reference values — cheap to pull from literature figures even without full tables)

| Metric | Target (Reτ=200, e.g. Moser–Kim–Mansour 1999 / Lee–Moser 2015) | Tolerance | Notes |
|---|---|---|---|
| Peak u'_rms+ | ≈ 2.6–2.7 near y+ ≈ 15 | ± 10% | Most commonly cited single-number check in papers — easy to eyeball off a plotted profile even without a data table. |
| Peak v'_rms+ | ≈ 0.8–0.9 near y+ ≈ 40–50 | ± 10–15% | Wall-normal fluctuation, more sensitive to resolution near the wall. |
| Peak w'_rms+ | ≈ 1.0–1.1 near y+ ≈ 20–30 | ± 10–15% | |
| Peak −⟨u'v'⟩+ (Reynolds shear stress) | plateau approaching 1 − y/δ trend, peak ≈ 0.65–0.7 | ± 10% | Should smoothly join the linear total-stress balance from §1. |
| Skewness/flatness of u' (optional, higher order) | qualitative shape match | — | Only worth doing if the lower-order moments already pass; sensitive to sample size. |

---

## 5. Domain-adequacy checks

| Metric | Target | Tolerance | Notes |
|---|---|---|---|
| Streamwise two-point correlation R_uu(Δx) at mid-channel or near-wall y+ | decays to ≈ 0 by Δx = Lx/2 | correlation < 0.1–0.2 at Lx/2 | Confirms Lx=12.57 (≈4πδ) is long enough not to artificially constrain large eddies. Classic overlooked check. |
| Spanwise two-point correlation R_uu(Δz) | decays to ≈ 0 by Δz = Lz/2 | correlation < 0.1–0.2 at Lz/2 | Lz=4.19 (≈4/3 πδ) is on the smaller side for Reτ=200 boxes reported in literature — this check matters more here than for Lx. If it doesn't decay, near-wall streak spacing may be artificially constrained. |

---

## 6. Resolution checks (grid adequacy, not accuracy per se — but explains any mismatches above)

| Metric | Target | Tolerance | Notes |
|---|---|---|---|
| Δx+ (streamwise) | ≲ 10–15 | — | With Lx=12.57 and 128 points: Δx+ = (Lx/128)·Re_τ ≈ 19.6 — on the coarse side for a fully resolved DNS; worth flagging explicitly if turbulence peak intensities are off. |
| Δz+ (spanwise) | ≲ 5–7 | — | With Lz=4.19 and 128 points: Δz+ ≈ 6.5 — reasonable. |
| Δy+ at wall (first point off wall) | ≲ 1 | — | Depends on your wall-normal clustering (129 points, presumably non-uniform e.g. via a stretching function) — compute directly from the grid. |
| Δy+ at centerline | ≲ 10–12 | — | Checks clustering isn't over-concentrated near walls at the expense of the core. |

If §1–4 show discrepancies beyond tolerance, check this table first — Δx+≈19.6 in particular is a plausible explanation for peak turbulence intensities (§4) or near-wall correlation lengths (§5) being off by more than the quoted tolerances.

---

## Suggested implementation order

1. §1 (residuals/BCs) — cheap, catches bugs before wasting time on physics comparisons.
2. §6 (resolution) — compute once, informs how tight your tolerances in §2–5 should realistically be.
3. §2 (Re checks) — cheap, high signal.
4. §3 (mean profile/log-law) — needs only universal constants.
5. §4 (turbulence statistics) — needs a couple of reference numbers, highest value once basics pass.
6. §5 (domain adequacy) — worth doing once, not a per-run check.

## Open items to confirm before filling in exact targets
- Whether incompact3d is running this case with constant flow rate (CFR) or constant pressure gradient (CPG) forcing — determines which of Re_τ / Re_b is the "free" check vs the derived one.
- Wall-normal grid stretching function/parameters (needed to compute Δy+ at wall and centerline in §6).
- Which reference dataset to anchor §4 against (MKM99 vs Lee & Moser 2015 vs another Reτ=200 DNS) — pick one primary source for consistency across all peak values.
