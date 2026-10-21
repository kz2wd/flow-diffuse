# Pressure Poisson solver (stretched `y`) — implementation plan

## Goal

A pressure Poisson solver on a **stretched `y` (wall) grid** that we **understand fully**,
replacing the old uniform-only `pressure_solve` (cell 50). It must pass:
1. a **uniform** analytical test (control), and
2. a **stretched** analytical test (INcompact3d `istret=2` grid, β=0.259, cell 64).

## The equation (sign fixed 2026-07-08)

```
    ∇²p = 2Q
```
**Not** `-∇²p = 2Q`. Derivation (incompressible NS, no body force, ρ=1):
- momentum `Du/Dt = -∇p + ν∇²u`; take divergence, use continuity → `∇²p = -[∂ᵢuⱼ ∂ⱼuᵢ]`.
- notebook defines `Q = -½ Σ ∂ᵢuⱼ ∂ⱼuᵢ` (cell 14), so `Σ ∂ᵢuⱼ ∂ⱼuᵢ = -2Q`.
- therefore `∇²p = -(-2Q) = 2Q`. ✓

Cross-checks:
- The old uniform solver (cell 50) builds a matrix whose main diagonal is the `∂yy` stencil
  plus `K = -(kx²+kz²)` — i.e. it inverts `∇²p = 2Q`. Consistent.
- Cell 64 uses `Q_exact = -3·p/2` for a test where `∇²p = -3p`: since `2Q = ∇²p = -3p`,
  `Q = -3p/2`. Consistent.

`∇² = ∂xx + ∂yy + ∂zz`, x/z periodic, y Neumann (two walls).

---

## Decision: Path B — our own clean solver (NOT an INcompact3d replica)

We initially tried to faithfully replicate INcompact3d's `poisson_010` (compact scheme +
"post-processing" basis rotations + pentadiagonal `A_y`). That failed and is **abandoned**.
Reasons (see "Why the INcompact3d-faithful route was dropped" below):
- The y "post-processing" (mirror-pair + `ay,by` phase, `poisson.f90` L761-780 / L923-944)
  is **not** a standalone invertible transform — its forward and inverse do not round-trip.
  It is machinery specific to INcompact3d's compact scheme + staggered layout, not a clean
  step in the abstract pressure solve. Replicating it half-understood was a recipe for failure.

**Path B** solves the clean 1D BVP directly, with a generic dense solver. No compact scheme,
no basis rotation, no fold/unfold, no pentadiagonal.

### The clean problem

After a plain FFT in x/z, each (x,z) mode `(i,k)` satisfies a 1D problem in `y`:
```
    (d²/dy² − λ) p = s,        λ = kx²[i] + kz²[k],   s = 2·Q̃[i,:,k]
```
Neumann BCs at both walls. This is a standard self-adjoint 1D BVP. (For the `(i=0,k=0)`
mode, λ=0 and `d²/dy²` has the constant nullspace → singular → skip it; it's the arbitrary
pressure constant.)

### Discretization (the part that must be right on a stretched grid)

Use the **conservative (flux-form) second derivative** — the standard choice for a
non-uniform grid, and it keeps the operator self-adjoint (symmetric matrix):
```
(p'')_j = ( 2 / (dy_j + dy_{j-1}) ) · [ (p_{j+1} − p_j)/dy_j  −  (p_j − p_{j-1})/dy_{j-1} ]
```
where `dy_j = y_j − y_{j-1}`. This is "apply d/dy twice, consistently."
- **Tridiagonal.** Off-diagonal coupling between `j` and `j+1` is the single number
  `2/(dy_j + dy_{j+1})`, appearing symmetrically → the matrix is **symmetric** (discrete
  self-adjointness of `d²/dy²` with Neumann BCs).
- **Walls (Neumann):** the end rows drop the ghost point → "half" stencils at `j=0` and
  `j=ny-1` (one fewer off-diagonal contribution).
- **Uniform limit:** reduces to the standard `(p_{j+1} − 2p_j + p_{j-1})/dy²` Laplacian.

### Pipeline
```
1.  S̃ = fftn(2Q, axes=(x,y,z))                    (plain DFT, no post-processing)
2.  precompute L_y (tridiagonal) from the y-grid — once, independent of (i,k)
3.  for each (i,k), skip (i=0,k=0):
        A = L_y − (kx²[i] + kz²[k])·I
        p̃[i,:,k] = np.linalg.solve(A, S̃[i,:,k])     (generic dense solver)
4.  p = ifftn(p̃, axes=(x,y,z))
5.  (optional) remove the y-mean to fix the pressure constant
```

`kx`, `kz` are **plain** wavenumbers `2π·fftfreq(n, d)` (not compact symbols) — Path B uses a
standard discretization, so plain spectral symbols are the correct x/z eigenvalues.

### Complexity
`ny ≈ 64`, `nx·nz ≈ 1024` modes → ~1024 dense solves of a 64×64 system. Trivial cost, and
no banded-format fragility (we hit `solve_banded` layout bugs on the old route).

---

## Tests

Test function (Neumann-compatible in y): `p = cos(2πx/Lx)·cos(πy/Ly)·cos(2πz/Lz)`.
Then `∇²p = −K2·p` with `K2 = (2π/Lx)² + (π/Ly)² + (2π/Lz)²`, so `2Q = −K2·p`, i.e.
**`Q = −(K2/2)·p`** (matches cell 64's `Q_exact = -3·p/2` for the `K2=3` case).

- **Uniform (control):** uniform y-grid. Expect `rel_rmse ~ 1e-10` (2nd-order accurate,
  well-resolved single mode).
- **Stretched (istret=2, β=0.259):** grid from `incompact_stretch` (cell 64). Expect
  `rel_rmse` **small** (a few ×1e-3), *not* 0.2. It will be less accurate than
  INcompact3d's 6th-order compact scheme (ours is 2nd-order in y), so the target is
  "clearly correct," not "1e-6." Optionally confirm with a **grid-convergence check**
  (2 resolutions, verify ~2nd-order convergence) to prove it's right rather than just small.

---

## Status / Next steps

- [ ] **Cell 1:** build `L_y` (flux-form tridiagonal) for a y-grid; verify symmetry, verify
      uniform limit = standard Laplacian, verify `L_y·cos(πy/Ly) ≈ −(π/Ly)²·cos(πy/Ly)` (1D).
- [ ] **Cell 2:** 1D solve `(L_y − λI)p = s` on a cosine, uniform + stretched, isolate y-operator.
- [ ] **Cell 3:** full 3D solve (FFT x/z → per-mode dense solve → IFFT x/z) + **uniform test** → ~1e-10.
- [ ] **Cell 4:** **stretched test** → small rel_rmse (+ optional convergence check).
- [ ] (Optional) if accuracy matters later, revisit a higher-order y-discretization — but
      only after the clean 2nd-order solver is verified.

---

## Appendix — Why the INcompact3d-faithful route was dropped (2026-07-08)

We spent a session trying to replicate `poisson_010` exactly (compact `wp` symbol, `yky`
eigenvalues, pentadiagonal `A_y`, and the y "DCT shift"). Findings:

1. **The y "DCT shift" is not a clean invertible step.** The forward (`poisson.f90` L777)
   maps a mirror-pair `(F[j], F[mirror])` → a value; the "inverse" (L940) does **not** undo
   it (round-trip fails, `idct(dct(F)) ≠ F`). It is specific to INcompact3d's compact scheme
   and staggered layout — not a general DFT→DCT conversion, and not verifiable in isolation.

2. **The equation sign is `∇²p = 2Q`, not `-∇²p = 2Q`.** Early plan cells used `-∇²`; the
   "sign-flipped" diagnostic result (`p_rec ≈ -p_exact`) was this convention mismatch, not a
   solver bug. (Corrected at the top of this file.)

3. **`build_full_A` already includes `kx²+kz²`;** adding `(kx²+kz²)·I` again double-counted
   and made every mode singular.

4. The compact-scheme basis ("post-processing" in x/z/y) is an artifact of the compact
   discretization, not a step in the abstract pressure solve. Replicating it without full
   understanding kept producing silent failures.

**Conclusion:** build our own clean solver (Path B) that we understand end-to-end. Higher
accuracy can be added later if needed.

### Key INcompact3d reference lines (for later, if we revisit compact scheme)
`poisson_010` (L663): fold L688-696, FFT L705, normalize L708-709, Z post L724-738,
X post L740-755, Y post L761-780, solve L821-897, zero-mean L904-906, Y unshift L923-944,
X unshift L960-975, Z unshift L977-991, IFFT L994, unfold L1000-1005.
`matrice_refinement` (L1921+), `waves` (L1530+), `abxyz` (L1480+), `inversion5_v1` (tools.f90 L1225).

## References
- `Incompact3d/src/poisson.f90` — the reference implementation (not being replicated, but the
  source of the grid + the sign convention cross-check).
- Notebook: `check_q-pressure.ipynb` — cell 14 (Q definition), cell 50 (old uniform solver),
  cell 64 (stretched grid `incompact_stretch` + `stretched_analytic_test`).
