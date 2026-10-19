# Stretched (INcompact3d-faithful) pressure solve — findings & implementation plan

## Goal

A pressure Poisson solver that is faithful to **INcompact3d** on a **stretched `y` (wall) grid**,
replacing the old uniform-only `pressure_solve`. It must pass:
1. a **uniform** analytical test (control), and
2. a **stretched** analytical test (INcompact3d `istret=2` grid, β=0.259).

Form:  `-∇²p = S = 2Q`,  `∇² = ∂xx + ∂yy + ∂zz`,  x/z periodic, y Neumann (two walls).

Old solver: `pressure.py::compute_pressure` and the notebook's `pressure_solve` (uniform-only,
stretched `rel_rmse ≈ 0.2`).

---

## What INcompact3d actually does (`src/poisson.f90 :: poisson_010`, istret=2)

`poisson_010` = Neumann in Y, periodic in X & Z (line 663). `ny = ny_global - 1`.

1. **Symmetry-fold the Y field** by even parity (lines 688-697): `rw2b(i,j)=rw2(i,2j-1)` and the
   mirror, so only `ny = ny_global-1` points are carried.
2. **FFT in X and Z** (periodic); normalize.
3. **Post-processing** multiplies modes by `az,bz/ax,bx` (periodic) and `ay,by`=sin/cos (the Y
   cosine/DCT basis) — INcompact3d's way of entering the compact-scheme basis.
4. **The Y solve** (the non-diagonal part — lines 853-880, the `istret /= 3` branch):
   ```fortran
   cw2 (i,j) = cw2b(i, 2*j-1)   ! odd  Y indices
   cw2c(i,j) = cw2b(i, 2*j)     ! even Y indices
   call inversion5_v1(a , cw2 )  ! pentadiagonal (5-band) solve, matrix a
   call inversion5_v1(a2, cw2c)  ! pentadiagonal (5-band) solve, matrix a2
   ! ... interleave odd/even back
   ```
   i.e. **two pentadiagonal solves per (x,z) mode** = the Y solve. `inversion5_v1`
   (`src/tools.f90` L1225) is a pentadiagonal Thomas-style solver.
   - (For `istret==0` it instead just divides by `kxyz(i,j,k)` from `waves()` — the diagonal case.)
5. **Backward post-processing** (Y, X, Z), IFFT back to physical.

### The matrix `a`/`a2` (built by `matrice_refinement()`, L1921+, istret=1/2 branch)

Main diagonal:
```
a(i,j) = -[ xk2(i)*transy(j)^2*transz(k)^2          ! X contribution
        + zk2(k)*transy(j)^2*transx(i)^2           ! Z contribution
        + xa0^2 * cw22(i,j)^2                       ! Y "uniform" part
        + xa1^2 * cw22(i,j)*(cw22(i,j-1)+cw22(i,j+1)) ]  ! Y stretched coupling
```
with `cw22(i,j) = transx(i)*yky(j)*transz(k)`, and for istret=2:
```
xa0 = alpha/pi + (1/2)/(beta*pi)
xa1 = -(1/4)/(beta*pi)
```
Off-diagonals (±1, ±2 → pentadiagonal): `xa01*cw22*(cw22±)` and `xa1^2*(-cw22(j±1)*cw22(j±2))`.

### The key realization
`xa0` and `xa1` **are the stretched metrics** `ppy`/`pp4y`. Recall
`ppy(η) = yly*(α/π + (1/(βπ))·sin²πη)` = a **constant + a cos(2πη) harmonic**. That single 2η
harmonic is *why* Y-mode `j` couples to `j±2` (pentadiagonal), and `xa1` (the cos/sin-2η
amplitude) is what generates the off-diagonals. So INcompact3d is doing exactly our **chain rule**:
```
d²/dy² = (1/ppy²)·∂ηη  −  (pp4y/ppy³)·∂η
         \______/     \____/
         diagonal    position-dependent -> a matrix in the Y-basis
```

### Coefficients (`src/schemes.f90` L891-899, the `...6` scheme)
```
2nd-deriv compact:  acix6*dx = 63/62,  bcix6*dx = 17/186,  alcaix6 = 9/62
compact symbol:     wp(w) = [2*(63/62)sin(w/2) + 2*(17/186)sin(3w/2)] / [1 + 2*(9/62)cos(w)]
                   xkx = nx*wp/xlx   (1st deriv);   xk2 = (nx*wp/xlx)^2  (2nd deriv = 1st^2)
```
Consistency check: `(a+3b)/(1+2α) = (63/62 + 3·17/186)/(1 + 2·9/62) = 1` exactly.
**`fhat'' = fhat'^2` is correct** (poisson.f90: `xk2 = xkx**2`) — do NOT "fix" it.

---

## One real difference vs. our chain-rule sketch

Our formula treats the directions as independent: `fhat2(kx) + d²/dy² + fhat2(kz)`.
INcompact3d instead carries **cross correction factors** `transx·transy·transz` (the compact
1st-deriv transfer function) on each term. For resolved (low) modes `trans ≈ 1`, so both reduce to
the same thing — it only matters near the Nyquist. We use the **independent chain-rule model**
(the user's own formula); add the `trans` coupling only if a test demands it.

---

## The implementation (cells 73-76 of `check_q-pressure.ipynb`)

### Architecture
1. `S = 2Q`.
2. DFT in X (axis 1) and Z (axis 3) — periodic.
3. Per (i,k) x/z mode, solve a **dense `ny×ny` system** in the Y direction:
   ```
   A(i,k) = (fhat2x(i) + fhat2z(k))·I + L_y
   L_y   = -(1/ppy²)·(∂ηη)  +  (pp4y/ppy³)·(∂η)     (chain rule)
   A·p_hat(i,:,k) = S_hat(i,:,k)
   ```
4. IFFT in X, Z; drop the Y-mean (null space).

### The subtle bug we hit (and fixed)
`∂η` maps cosine (Neumann) modes to **sine** modes — it is **not diagonal** in the pure DCT basis.
A naive "orthonormalize the DCT matrix and use it as a diagonal basis" is **wrong**, because the
cosine eigenfunctions of `∂ηη` are only orthogonal under the **weighted** (DCT) inner product
(`w_0 = w_{ny-1} = 1`, else `2`), not the unweighted one.

**Correct construction** (in `eta_operators`): build the operators in the uniform `η`-index grid
(0..1) using the weighted-basis projection
```
analysis:  f_hat = diag(1/norm) @ Bcos @ diag(w) @ f
D2 = Bcos.T @ diag(fhat2) @ analysis     # ∂ηη : cos -> cos
D1 = Bsin.T @ diag(fhat1) @ analysis     # ∂η  : cos -> sin
   fhat1 = -(1/dη)·wp(ω)                 # d/dη  cos_m -> -ω sin_m
   fhat2 = -(1/dη)²·wp(ω)²               # ∂ηη cos_m eigenvalue (negative)
   ω_m   = π·m/(ny-1),  dη = 1/(ny-1)
```
This passed analytic-derivative verification (cell 74):
```
d/deta   cos(πη)   rel err = 2.4e-12
d2/deta2 cos(πη)   rel err = 7.6e-12
d2/deta2 cos(5πη)  rel err = 7.5e-8   (6th-order compact, mode 5)
```

### Solver details
- X/Z compact 2nd-deriv symbols: `fhat2x(i) = (wp(kx_i·Lx)/Lx)²`, `fhat2z` likewise
  (`kx = 2π·fftfreq(nx, d=Lx)`).
- Stretched `L_y = -diag(1/ppy²)@D2 + diag(pp4y/ppy³)@D1`.
- Loop: `for i in nx, k in nz: A = (fhat2x[i]+fhat2z[k])*I + L_y; P[:,i,:,k] = solve(A, S_hat[:,i,:,k]).T`.

### Grid / metrics
- Stretched grid + `ppy` from `incompact_stretch` (cell 64, verified `y[0]=0, y[-1]=yly=π`).
- `pp4y = d(ppy)/dη` computed **numerically** as `np.gradient(ppy, 1/(ny-1))` — robust and
  consistent with the actual grid (avoids an analytic-formula mismatch).
- Uniform control: `ppy = yly` (const), `pp4y = 0`.
- Test function (Neumann-compatible): `p = cos(2πx/Lx)·cos(πy/Ly)·cos(2πz/Lz)`;
  `-∇²p = K2·p`, `K2 = (2π/Lx)²+(π/Ly)²+(2π/Lz)²`; set `S = K2·p` (so `Q = S/2`).

---

## Status

- **Operators** (`eta_operators`): **VERIFIED** against analytic derivatives (cell 74, above).
- **`pressure_solve_v3`** (cells 73-76): written and fixed two bugs —
  1. axis mix-up (`Shat[:,i,:,k]` is the Y-vector for mode (i,k), not `Shat[i,:,k]`),
  2. test passed 3D where the solver expects batched 4D (`0.5*S[None,...]`).
- **Uniform + stretched `rel_rmse`: NOT yet captured.** The last `execute_cell(76)` hit a
  **Jupyter-MCP "server shut down"** (client-side streamable-http session drop; the jupyter backend
  on :8888 was healthy — 200 on `/api`). The test numbers are still to be recorded.

## Next steps
1. Re-establish the Jupyter-MCP session, then `execute_cell(76)` to record
   `rel_rmse` (uniform should be ~1e-8; stretched should drop from ~0.2 to ~1e-6).
2. If the stretched error is still high, the likely gaps are: (a) the missing INcompact3d
   `transx·transy·transz` coupling (only matters near Nyquist — unlikely for this low-mode test), or
   (b) the exact `pp4y` metric convention — cross-check against `src/stretching.f90`.
3. (Optional, perf) Replace the dense per-mode solve with INcompact3d's even/odd split into two
   pentadiagonal `solve_banded((2,2))` systems (see the PERF NOTE in cell 73 and the
   `matrice_refinement`/`inversion5_v1` code) — only if performance ever matters.

## References
- `Incompact3d/src/poisson.f90` — `poisson_010` (L663), `matrice_refinement` (L1814+), `waves` (L1530).
- `Incompact3d/src/schemes.f90` — `...6` coefficients (L891-931).
- `Incompact3d/src/tools.f90` — `inversion5_v1` (L1225, pentadiagonal solver).
- `pressure.py::compute_pressure` — the existing banded-in-Y-per-(x,z)-mode pattern we generalized.
- Notebook: `check_q-pressure.ipynb` cells 64 (grid), 73-76 (new solver + test).
