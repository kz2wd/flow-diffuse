"""Reusable statistical evaluation of a predicted field vs DNS.

Produces a flat dict of scalars + profiles (1-D arrays along y) + spectra,
so results can be:
  - printed,
  - saved as JSON/npz "intermediate data" for a later comparison web page,
  - re-ranked without retraining.

All metrics operate on fields of shape (N, 3, X=24, Y=33, Z=24) in PHYSICAL
units, plus the y-coordinate array.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew as _skew
from scipy.stats import kurtosis as _kurtosis

NX, NY, NZ = 24, 33, 24
LX, LZ = 1.755 * np.pi, 1.2 * np.pi
COMP = ['u_x', 'u_y', 'u_z']

# wavenumber grid (shared)
_KX = np.fft.fftfreq(NX, d=LX / NX) * 2 * np.pi
_KZ = np.fft.fftfreq(NZ, d=LZ / NZ) * 2 * np.pi
_KXM, _KZM = np.meshgrid(_KX, _KZ, indexing="xy")
K_MAG = np.sqrt(_KXM ** 2 + _KZM ** 2)

# default annuli
DK = 1.0
K_EDGES = np.arange(0.0, K_MAG.max() + DK, DK)
K_CENTERS = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
ANN_MASKS = [(K_MAG >= K_EDGES[i]) & (K_MAG < K_EDGES[i + 1])
             for i in range(len(K_CENTERS))]
NONEMPTY = np.array([m.sum() > 0 for m in ANN_MASKS])
K_CENTERS_P = K_CENTERS[NONEMPTY]


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _mean_std_y(field):
    """mean/std over (N, X, Z) -> (C, Y)."""
    return field.mean(axis=(0, 2, 4)), field.std(axis=(0, 2, 4))


def _fluct(field):
    return field - field.mean(axis=(0, 2, 4), keepdims=True)


def _energy_spectrum(field, comp):
    F = np.fft.fft2(field[:, comp], axes=(1, 3))
    return (np.abs(F) ** 2) / (NX * NZ)


def _spectral_energy_ek(field):
    """E(k) per component, per non-empty annulus -> (n_bins, 3)."""
    E = np.zeros((len(K_CENTERS), 3))
    for c in range(3):
        spec = _energy_spectrum(field, c)
        for i, m in enumerate(ANN_MASKS):
            E[i, c] = (spec * m[None, :, None, :]).sum(axis=(1, 3)).sum()
    return E[NONEMPTY]


def _spectral_energy_ek_full(spec):
    """E(k) per non-empty annulus from a precomputed spec (3,N,X,Y,Z) -> (n_bins,3)."""
    E = np.zeros((len(K_CENTERS), 3))
    for c in range(3):
        for i, m in enumerate(ANN_MASKS):
            E[i, c] = (spec[c] * m[None, :, None, :]).sum(axis=(1, 3)).sum()
    return E[NONEMPTY]


def _all_spectra(field):
    """(3, N, X, Y, Z) spectrum for all components (computed once)."""
    return np.stack([_energy_spectrum(field, c) for c in range(3)])


def _rms_reynolds(field):
    uf = _fluct(field)
    rms = np.sqrt((uf ** 2).mean(axis=(0, 2, 4)))                 # (C, Y)
    R = np.einsum("n c i y z, n d i y z -> c d y", uf, uf)          # (C,C,Y)
    return rms, R


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def evaluate(pred: np.ndarray, target: np.ndarray, y: np.ndarray) -> dict:
    """Compute all metrics. pred/target: (N,3,24,33,24) physical units.

    Returns a dict of scalars and 1-D arrays (along y or |k|). All values are
    JSON-serializable (floats / lists).
    """
    out = {"n_samples": int(pred.shape[0])}

    # ---- global / per-component MSE ----
    out["mse_global"] = float(((pred - target) ** 2).mean())
    out["mse_per_comp"] = {COMP[c]: float(((pred - target)[:, c] ** 2).mean())
                           for c in range(3)}
    out["ev_global"] = float(1.0 - ((pred - target) ** 2).mean()
                             / target.var())

    # ---- mean & std profiles along y ----
    mean_p, std_p = _mean_std_y(pred)
    mean_t, std_t = _mean_std_y(target)
    out["mean_pred"], out["mean_tgt"] = mean_p.tolist(), mean_t.tolist()
    out["std_pred"], out["std_tgt"] = std_p.tolist(), std_t.tolist()
    out["max_abs_dmean"] = float(np.abs(mean_p - mean_t).max())
    out["max_abs_dstd"] = float(np.abs(std_p - std_t).max())

    # ---- MSE profile along y ----
    out["mse_y"] = (((pred - target) ** 2).mean(axis=(0, 2, 4))).tolist()

    # ---- RMS & Reynolds ----
    rms_p, R_p = _rms_reynolds(pred)
    rms_t, R_t = _rms_reynolds(target)
    out["rms_pred"], out["rms_tgt"] = rms_p.tolist(), rms_t.tolist()
    out["reynolds_pred"] = R_p.tolist()
    out["reynolds_tgt"] = R_t.tolist()
    out["max_abs_drms"] = float(np.abs(rms_p - rms_t).max())
    # shear stresses of interest
    out["reynolds_uv_pred"] = R_p[0, 1].tolist()
    out["reynolds_uv_tgt"] = R_t[0, 1].tolist()

    # ---- spectrum E(k) (compute each spectrum ONCE, reuse) ----
    spec_p = _all_spectra(pred)
    spec_t = _all_spectra(target)
    Ek_p = _spectral_energy_ek_full(spec_p)
    Ek_t = _spectral_energy_ek_full(spec_t)
    out["k_centers"] = K_CENTERS_P.tolist()
    out["Ek_pred"] = Ek_p.tolist()
    out["Ek_tgt"] = Ek_t.tolist()

    # relative spectral error per annulus (MSE / target energy) — reuse spec
    rel = np.full((len(K_CENTERS_P), 3), np.nan)
    for c in range(3):
        mse_E = np.array([
            (((spec_p[c] - spec_t[c]) * m[None, :, None, :]) ** 2).sum()
            for m in ANN_MASKS])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel[NONEMPTY, c] = np.where(
                Ek_t[:, c] > 0, mse_E / np.maximum(Ek_t[:, c], 1e-300), np.nan)
    out["rel_spectral_err"] = rel.tolist()

    # fraction of model energy above the DNS resolved range (DNS ~ zero there)
    # find first annulus where target E(k) (u_x) drops below 1e-6 of its max
    ref = Ek_t[:, 0]
    if ref.max() > 0:
        thr = 1e-6 * ref.max()
        resolved = int(np.max(np.where(ref > thr, np.arange(len(K_CENTERS_P)), 0)))
        k_resolved = float(K_CENTERS_P[resolved])
    else:
        k_resolved = float(K_CENTERS_P[-1])
    out["k_resolved_est"] = k_resolved
    # per-comp fraction of pred energy above k_resolved
    f = {}
    for c in range(3):
        above = Ek_p[:, c][K_CENTERS_P > k_resolved].sum()
        f[COMP[c]] = float(above / max(Ek_p[:, c].sum(), 1e-300))
    out["frac_energy_above_kresolved"] = f

    # ---- PDF shape stats at channel center ----
    yc = NY // 2
    sk, ku = {}, {}
    for c in range(3):
        vp = _fluct(pred)[:, c, :, yc, :].ravel()
        vt = _fluct(target)[:, c, :, yc, :].ravel()
        sk[COMP[c]] = [float(_skew(vt)), float(_skew(vp))]
        ku[COMP[c]] = [float(_kurtosis(vt)), float(_kurtosis(vp))]
    out["skew_center"] = sk
    out["kurtosis_center"] = ku

    return out


# ---------------------------------------------------------------------------
# persistence (intermediate data for the later web page)
# ---------------------------------------------------------------------------
def to_jsonable(d: dict):
    """Ensure numpy -> python for JSON."""
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o
    return conv(d)


def save_results(res: dict, path: str):
    import json
    with open(path, "w") as f:
        json.dump(to_jsonable(res), f, indent=2)
    return path


def load_results(path: str) -> dict:
    import json
    with open(path) as f:
        return json.load(f)
