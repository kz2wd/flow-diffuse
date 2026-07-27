# FlowDiffuse — Project Design Document

## Problem Statement

Reconstruct 3D velocity fields **u(x,y,z,t)** from wall pressure measurements **p_wall(x,z,t)**. The flow domain has spatial dimensions typically 64–128+ in each direction, with data stored as zarr arrays on a MinIO server (~10k–20k training samples).

## Core Architectural Decisions

### 1. Two-Stage Reconstruction Pipeline

```
                    Stage 1: Analytical (GREEN'S FUNCTION)
                    ─────────────────────────────────────
                    wall pressure p_wall(x,z,t) + confidence c(x,z,t)
                              │
                        Green's integral:
                      v₀ = ∫_wall G(x,y,z; x',z') · p_wall(x',z') dx'dz'
                              │
                       → Rough field v_guess(y)
                       → Initial uncertainty σ₀(y)

                    Stage 2: Multi-Scale Diffusion (NEURAL CORRECTION)
                    ────────────────────────────────────────────────────
                    v_guess + noise ← init diffusion from analytical guess
                              │
                    For each denoising step k:
                        a. Decompose into fixed frequency bands
                        b. UNet predicts turbulence correction per scale
                        c. Project onto physics constraints
                        d. Update confidence field
                              │
                       → Reconstructed v(x,y,z,t) + σ(x,y,z,t)
```

**Why Green's function first**: It gives the network a physically valid starting point. The diffusion doesn't need to learn Navier-Stokes from scratch — it just needs to learn *"what the analytical propagation missed because of turbulence/nonlinearity."* This drastically reduces the required number of sampling steps and stabilizes training.

### 2. Multi-Scale Decomposition (Fixed, Not Learned)

The network does **not** decide what scales matter. We do that explicitly via bandpass decomposition:

```
Full field v[y,x,z] 
       │
       ▼
┌─────────────────────────────────────┐
│  Discrete Wavelet Transform (DWT)  │  ← happens BEFORE the UNet, on CPU
│  or log-spaced frequency filterbank │     no learnable parameters here
└─────────────────────────────────────┘
       │
   ┌───┼───┐───┬───►
   ▼   ▼   ▼   ▼
 Scale0 Scale1 Scale2 Scale3        ← each branch gets its own UNet head
 coarse      fine (near-wall)        or shared weights across scales

Each scale is processed independently by a UNet block, then recombined.
```

Concrete implementation: **Discrete Wavelet Transform** (Haar or Daubechies-2) along the wall-normal axis `y`. This cleanly separates near-wall high-frequency structures from far-field low-frequency ones without any learned parameters.

| Scale | Physical meaning | UNet attention focus |
|-------|------------------|---------------------|
| Coarsest | Far-field bulk flow, recirculation | Global structure, integral constraints |
| Mid-coarse | Large vortical structures | Vortex coherence, circulation |
| Mid-fine | Shear layer dynamics | Momentum exchange |
| Finest | Near-wall streaks/turbulence | Local BC matching, boundary layer |

Each scale gets its own branch (not a single multi-scale block) — so the network processes scales in parallel but can learn different corrections at each level. Final field = IWT (inverse wavelet transform) of all branches.

### 3. Spatial-Temporal Kernel Patching (No Full-Sequence Attention)

This is the key insight for tractability:

```
We don't reconstruct the full [T, D, H, W] at once.
We reconstruct a SLIDING PATCH of size [t_win, d_patch, h_patch, w_patch].
Each patch gets conditioned on its neighbors (overlap region).

Patch topology during inference:

Time axis (sliding):
──────────────────────→ t
[patch_1][patch_2_overlap][patch_3_overlap]...

At each timestep within the window:
        y
        │   ┌─────────────────┐
    D   │   │    flow field   │  ← patch center: unknown, to be reconstructed
        │   │ (UNet input)    │
        │   └──────┬──────────┘
            known│BC at y=0 wall from pressure
           Dirichlet│at y=D boundary from symmetry/zero
       ┌────────────┴────────────┐
       x  ◄─── h_patch ───► z     ← lateral dimensions

Neighboring patches provide boundary conditions on the patch edges.
Temporal neighbors (past/future windows) provide time continuity constraints.
```

### Patch Management During Inference

```python
class KernelReconstruction:
    """Manages overlapping patches for spatial-temporal reconstruction."""
    
    def reconstruct_sequence(self, pressure_field, confidence_field):
        """Given full pressure timeline, reconstruct velocity field patch by patch."""
        
        results = {}
        overlap_counters = {}
        
        # 1. Run diffusion on each patch independently (parallelizable)
        for patch_id in all_patches:
            p_wall_patch = extract_pressure_patch(pressure_field, patch_id)
            c_conf_patch = extract_confidence_patch(confidence_field, patch_id)
            
            # Extract boundary conditions from neighboring patches
            bc_left = get_neighbor(results, patch_id, "left")
            bc_right = get_neighbor(results, patch_id, "right")
            bc_past = get_neighbor(results, patch_id, "t_minus_1")
            bc_future = get_neighbor(results, patch_id, "t_plus_1")
            
            v_patch, sigma_patch = self.denoise_patch(
                pressure=p_wall_patch,
                confidence=c_conf_patch,
                boundary_conditions={
                    "wall": p_wall_patch,      # y=0: Dirichlet from pressure
                    "left_right": [bc_left, bc_right],
                    "temporal": [bc_past, bc_future]
                }
            )
            results[patch_id] = v_patch
            
            # Accumulate into overlap regions (weighted average)
            self.add_to_overlaps(results[patch_id], patch_id)
        
        return stitch_patches(results), compute_final_confidence(overlap_counters)
```

During **training**, patches are extracted randomly from the full zarr array — each batch is a random spatiotemporal patch. This means the model never needs to see more than [t_win, d_patch, h_patch, w_patch] at once, dramatically reducing VRAM.

### Why This Works for Turbulence

- **Physical locality**: Turbulent structures have finite correlation lengths. A patch covering a few integral length scales is enough — you don't need the full domain.
- **Neighborhood conditioning**: Overlap regions naturally enforce smoothness without global constraints.
- **Parallelizable**: Each patch can be processed independently during inference (unlike RNN/sequence models).

### 4. Constraint Enforcement (Simple, Robust)

Two constraints, both straightforward to apply per denoising step:

#### A. Incompressibility — Helmholtz-Hodge Projection

```
After each denoising step produces candidate v*:
    div(v*) = f(x,y,z,t)   ← generally not zero
    
    Solve Poisson equation: ∇²φ = f
        → φ = ∇⁻²(div(v*))     (FFT-based or conjugate gradient solver)
    
    Correct: v_corrected = v* - ∇φ
    
    This guarantees div(v_corrected) = 0 exactly (up to numerical precision).
```

This is a **mathematically exact** projection — no learning needed. It's essentially what the network does *after* each UNet step, not part of the UNet itself. Works on patches with zero-Neumann BC at boundaries.

#### B. Boundary Condition — Direct Enforcement

```
At wall (y=0): v|_wall = 0 (no-slip) or v_tangential from pressure via NS momentum eq
    → Just overwrite the boundary values directly after each step.

At far-field (y=D): v → free-stream value or symmetry
    → Same: direct overwrite.

No projection needed — Dirichlet BCs are enforced by replacement.
```

**Constraint ordering during each denoising step:**

1. Denoise (UNet prediction)
2. Overwrite wall/far-field BC (Dirichlet → direct replace)
3. Helmholtz-Hodge project (→ enforce div = 0)

Order doesn't matter much because:
- Step 2 is exact replacement (hard constraint)
- Step 3 makes the field divergence-free regardless of what step 1/2 produced
- The UNet learns to predict in a space that, when projected, gives valid turbulence — it adapts

### 5. Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                              │
└─────────────────────────────────────────────────────────────────────┘

PRESSURE INPUT                     GREEN'S PROPAGATOR              DIFFUSION PATCHES
[128, H, W, T]                    [analytical integration]        [sliding kernel reconstruction]

     │                                    │                              │
     ▼                                    ▼                              ▼
  p_wall ──→ Green's integral ──→ v_guess(y) → patch extract ──┐
  c_conf ──→ weighting func.    → σ₀(y)         │              │
                                                    │              │
                                            ┌───────┴──────────────┴───────┐
                                            │       OVERLAPPING PATCHES      │
                                            │                                │
                                            │  patch_A: UNet step + BC +     │
                                            │             Hodge projection    │
                                            │  patch_B: independent, parallel │
                                            │  ...                           │
                                            └───────┬──────────────┬───────┘
                                                    │              │
                                          stitch overlap   weighted σ merge
                                                    │              │
                                                    ▼              ▼
                                              stitched field     final σ


┌─────────────────────────────────────────────────────────────────────┐
│  EACH PATCH: MULTI-SCALE UNET CORE                                   │
└─────────────────────────────────────────────────────────────────────┘

   noisy_patch + pressure_patch + confidence_patch
                │
        ┌───────┼──────────┐
        ▼       ▼          ▼
     Scale0  Scale1      Scale2      ← DWT along wall-normal axis y
   (coarse) (mid)      (fine)       ← fixed decomposition, no learnable params
        │       │          │
    UNet_blk UNet_blk   UNet_blk   ← shared weights across scales
        │       │          │
        └───────┼──────────┘
                ▼
           IWT combine
                ▼
        turbulence_correction_pred (noise)


┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING: RANDOM PATCH EXTRACTION                                   │
└─────────────────────────────────────────────────────────────────────┘

  From zarr on MinIO (lazy):
     pressure[t:y+h:x+w]  → torch.Tensor [t_win, d_patch, h_patch, w_patch]
     
  Each batch = N random spatiotemporal patches from training set
  
  Loss per patch:
     L = MSE(v_pred_versus_v_target) 
       + λ_div * ||div(v_stitched)|²  (enforced exactly via Hodge projection)
       + λ_bc * BC_violation_on_walls  (enforced by direct overwrite, residual on momentum eq)
```

---

## File Structure

```
flow-diffuse/
├── pyproject.toml                    # deps: torch, diffusers>=0.27, xarray, zarr, s3fs, 
│                                      einops, hydra-core, tyro, wandb, PyWavelets (for fixed DWT)
├── configs/default.yaml              
├── src/flow_diffuse/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # MinIOZarrStore + PressureFlowDataset
│   │   └── transforms.py            # ConfidenceMasking, NormalizePressure
│   ├── model/
│   │   ├── __init__.py
│   │   ├── propagator.py            # Green's function wall→interior analytically
│   │   ├── multiscale.py            # DWT decomposition + IWT recombination
│   │   ├── backbone.py              # Multi-scale UNet3DModel with per-scale heads
│   │   ├── diffusion/
│   │   │   ├── __init__.py
│   │   │   ├── scheduler.py         # DualDiffusionSchedule (wall-normal + time)
│   │   │   └── sampler.py           # ConstrainedSampler: BC overwrite → Hodge proj
│   │   └── loss.py                  # MSE + divergence residual losses
│   ├── reconstruction/              # NEW: kernel patching layer
│   │   ├── __init__.py
│   │   └── kernel_recon.py          # KernelReconstruction class
│   ├── train.py                     # Training loop (random patches, EMA, wandb)
│   └── utils/
│       ├── __init__.py
│       ├── physics.py               # Divergence ops, Hodge projection, BC operators
│       └── metrics.py               # Reconstruction error, spectral analysis, BC violation
├── tests/                           # pytest with fake tensors
└── scripts/sample.py                # Standalone inference using kernel recon
```

---

## Remaining Open Questions

1. **Green's function specifics**: Which Green's function form is applicable for your flow regime (Stokeslet? Oseenlet? Point source)? Need to verify the analytical integral setup matches your boundary conditions.
2. **Patch size selection**: What are the characteristic integral length scales of your flow? This determines the minimum patch dimensions and overlap width needed for accurate stitching.
3. **Temporal windowing**: How long do turbulent structures persist relative to the sampling rate? Determines optimal `t_win` for kernel patches.
