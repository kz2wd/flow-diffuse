# Project context

## What this is
Personal scientific research project, not production software. Goal:
reconstruct the full 3D instantaneous velocity field of wall-bounded turbulent
channel flow from sparse wall-pressure measurements, using a physics-guided
diffusion model. Sits at the intersection of generative ML (diffusion models,
`diffusers`) and computational fluid mechanics (incompressible Navier-Stokes,
DNS turbulence data).

Design is under active iteration — treat the choices below as current working
decisions being empirically benchmarked, not settled conclusions. Don't assume
they're final just because they're written down.

## Current design (see `diffusion_flow_reconstruction_spec.md` for full detail)
- **State representation:** primary candidate is `(v, ω_y)` (Kim–Moin wall-normal
  velocity/vorticity formulation) — exactly invertible to full velocity,
  continuity satisfied by construction. Benchmarking against toroidal–poloidal
  potentials, vector vortex field, POD/Galerkin coefficients, and Q/λ2/Δ/λ_ci
  invariants (diagnostic-only — used as guidance terms, not as state, since they
  aren't invertible on their own).
- **Multi-scale:** coarse-to-fine diffusion cascade + spectral SNR equalization
  (power-law noise schedule), to counter turbulence's power-law spectrum being
  corrupted unevenly by standard noise schedules.
- **Physics guidance at inference time:** no-slip/no-penetration BC, NS momentum
  + continuity residual, pressure–Poisson consistency (`∇²p = 2ρQ`) tying the
  current estimate back to the measured wall pressure.
- **Base implementation:** HuggingFace `diffusers` (`UNet3DConditionModel` or
  custom 3D UNet + scheduler), custom sampling loop — guidance runs inside the
  loop, so no prebuilt `DiffusionPipeline`.

## How I want you to work with me
I'm the lead developer on this project and I'm building my own competence in it
deliberately — that's a goal in itself, not just a means to the result. Default to:

- **Guidance and co-programming, not autonomous implementation.** Explain the
  reasoning behind an approach before or instead of writing the full thing.
  Prefer small, reviewable increments I implement myself over large generated
  blocks I'd just accept.
- **Ask before big decisions.** Architecture changes, new dependencies, or
  swapping a core design choice (state representation, scheduler, guidance
  formulation) get flagged and discussed, not made silently.
- **Teach, don't just solve.** If there's a standard technique, pitfall, or piece
  of domain background I should know for next time, say so explicitly, even if
  it slows the immediate task down.
- **I'll ask directly when I do want you to just write something** — boilerplate,
  a function I've already designed, a plotting/utility script. Otherwise assume
  I want to write it myself with your input, not receive it finished.

## Domain notes
- Assume DNS/CFD terminology (wall units, Re_τ, near-wall vs. outer layer,
  Kolmogorov spectrum) without re-explaining unless asked.
- Assume familiarity with diffusion model internals (score/noise prediction,
  DDPM/DDIM, classifier-free/posterior-sampling guidance) without re-deriving
  basics unless asked.
