#!/usr/bin/env python3
"""
Visualization utility for Ret_tau=180 channel DNS data.

Expected directory layout when this script is placed in the working directory:

  ./param.txt
  ./yp.dat
  ./Ret180/vel/ux200
  ./Ret180/vel/uy200
  ./Ret180/vel/uz200
  ./Ret180/pres/p_rapid200
  ./Ret180/pres/p_slow200
  ./Ret180/pres/p_stokes200
  ./Ret180/QR/Q200
  ./Ret180/QR/R200

Assumptions:
  - Binary fields are little-endian float64 (double).
  - Data are stored with x as the fastest varying index, then y, then z.
  - Loaded arrays are returned as a[x, y, z] with shape (Nx, Ny, Nz).

Examples:
  python visualize_ret180.py --quick
  python visualize_ret180.py --field ux --plane xz --j 10
  python visualize_ret180.py --field p_total --plane xy --k 64
  python visualize_ret180.py --field Q --plane yz --i 128
  python visualize_ret180.py --profile ux
  python visualize_ret180.py --qr-map
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt


Plane = Literal["xy", "xz", "yz"]


@dataclass(frozen=True)
class Params:
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float


def read_params(path: Path = Path("param.txt")) -> Params:
    """Read Nx, Ny, Nz, Lx, Ly, Lz from the first six lines of param.txt."""
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(float(s))
    if len(vals) < 6:
        raise ValueError(f"{path} must contain at least 6 numeric values: Nx, Ny, Nz, Lx, Ly, Lz")
    return Params(int(vals[0]), int(vals[1]), int(vals[2]), vals[3], vals[4], vals[5])


def read_y(path: Path = Path("yp.dat"), params: Params | None = None) -> np.ndarray:
    """Read wall-normal grid positions."""
    y = np.loadtxt(path, dtype=np.float64)
    if params is not None and y.size != params.Ny:
        raise ValueError(f"{path} contains {y.size} points, but Ny={params.Ny}")
    return y


def field_path(field: str, time: str) -> Path | list[Path]:
    """Return the path(s) corresponding to a named field."""
    field = field.lower()
    if field in {"ux", "uy", "uz"}:
        return Path("Ret180") / "vel" / f"{field}{time}"
    if field == "q":
        return Path("Ret180") / "QR" / f"Q{time}"
    if field == "r":
        return Path("Ret180") / "QR" / f"R{time}"
    if field in {"p_rapid", "rapid"}:
        return Path("Ret180") / "pres" / f"p_rapid{time}"
    if field in {"p_slow", "slow"}:
        return Path("Ret180") / "pres" / f"p_slow{time}"
    if field in {"p_stokes", "stokes"}:
        return Path("Ret180") / "pres" / f"p_stokes{time}"
    if field in {"p", "pressure", "p_total", "total"}:
        return [
            Path("Ret180") / "pres" / f"p_rapid{time}",
            Path("Ret180") / "pres" / f"p_slow{time}",
            Path("Ret180") / "pres" / f"p_stokes{time}",
        ]
    raise ValueError(
        f"Unknown field: {field}. Use ux, uy, uz, Q, R, p_rapid, p_slow, p_stokes, or p_total."
    )


def read_binary_field(path: Path, params: Params) -> np.ndarray:
    """
    Read one little-endian double field and return array a[x, y, z].

    Storage assumption: x changes fastest, then y, then z. Therefore the raw
    one-dimensional array is first reshaped as (Nz, Ny, Nx), then transposed.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    n_expected = params.Nx * params.Ny * params.Nz
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != n_expected:
        raise ValueError(
            f"{path}: expected {n_expected} doubles from Nx*Ny*Nz "
            f"= {params.Nx}*{params.Ny}*{params.Nz}, but got {raw.size}. "
            "Check dimensions, precision, endian, or index order."
        )
    return raw.reshape((params.Nz, params.Ny, params.Nx), order="C").transpose(2, 1, 0)


def load_field(field: str, params: Params, time: str = "200") -> np.ndarray:
    """Load a scalar field. For p_total, sum rapid, slow, and stokes pressure."""
    p = field_path(field, time)
    if isinstance(p, list):
        arr = np.zeros((params.Nx, params.Ny, params.Nz), dtype=np.float64)
        for pp in p:
            arr += read_binary_field(pp, params)
        return arr
    return read_binary_field(p, params)


def make_coordinates(params: Params, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create x, y, z coordinate arrays from Lx, Ly, Lz and yp.dat."""
    x = np.linspace(0.0, params.Lx, params.Nx, endpoint=False)
    z = np.linspace(0.0, params.Lz, params.Nz, endpoint=False)
    return x, y, z


def default_index(params: Params, axis: str) -> int:
    if axis == "x":
        return params.Nx // 2
    if axis == "y":
        return params.Ny // 2
    if axis == "z":
        return params.Nz // 2
    raise ValueError(axis)


def slice2d(a: np.ndarray, plane: Plane, i: int | None, j: int | None, k: int | None, params: Params):
    """Return 2D data and labels for a selected plane."""
    if plane == "xy":
        kk = default_index(params, "z") if k is None else k
        return a[:, :, kk].T, "x", "y", f"z-index k={kk}"
    if plane == "xz":
        jj = default_index(params, "y") if j is None else j
        return a[:, jj, :].T, "x", "z", f"y-index j={jj}"
    if plane == "yz":
        ii = default_index(params, "x") if i is None else i
        return a[ii, :, :].T, "y", "z", f"x-index i={ii}"
    raise ValueError(plane)


def plane_extent(plane: Plane, x: np.ndarray, y: np.ndarray, z: np.ndarray, params: Params):
    if plane == "xy":
        return [x[0], params.Lx, y[0], y[-1]]
    if plane == "xz":
        return [x[0], params.Lx, z[0], params.Lz]
    if plane == "yz":
        return [y[0], y[-1], z[0], params.Lz]
    raise ValueError(plane)


def symmetric_limits(data: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    """Robust symmetric color limits, useful for fluctuating quantities."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    m = np.percentile(np.abs(finite), percentile)
    if m == 0:
        m = np.max(np.abs(finite)) or 1.0
    return -m, m


def robust_limits(data: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = np.percentile(finite, 100.0 - percentile)
    hi = np.percentile(finite, percentile)
    if lo == hi:
        lo = np.min(finite)
        hi = np.max(finite)
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def plot_slice(
    field: str,
    params: Params,
    y: np.ndarray,
    time: str,
    plane: Plane,
    i: int | None,
    j: int | None,
    k: int | None,
    outdir: Path,
    show: bool,
    cmap: str,
    symmetric: bool,
    dpi: int,
) -> Path:
    a = load_field(field, params, time=time)
    x, yy, z = make_coordinates(params, y)
    s, xlabel, ylabel, subtitle = slice2d(a, plane, i, j, k, params)
    extent = plane_extent(plane, x, yy, z, params)

    if symmetric:
        vmin, vmax = symmetric_limits(s)
    else:
        vmin, vmax = robust_limits(s)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{field}_{time}_{plane}_{subtitle.replace(' ', '_').replace('=', '')}.png"

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    im = ax.imshow(
        s,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(field)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{field}, {plane}-slice, {subtitle}, time={time}")
    fig.savefig(out, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_profile(
    field: str,
    params: Params,
    y: np.ndarray,
    time: str,
    outdir: Path,
    show: bool,
    dpi: int,
) -> Path:
    a = load_field(field, params, time=time)
    prof = a.mean(axis=(0, 2))

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"profile_{field}_{time}.png"

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    ax.plot(prof, y, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel(f"<{field}>_xz")
    ax.set_ylabel("y")
    ax.set_title(f"Plane-averaged profile: {field}, time={time}")
    ax.grid(True, linewidth=0.5)
    fig.savefig(out, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_qr_map(
    params: Params,
    time: str,
    outdir: Path,
    show: bool,
    sample: int,
    seed: int,
    dpi: int,
) -> Path:
    q = load_field("Q", params, time=time).ravel()
    r = load_field("R", params, time=time).ravel()

    finite = np.isfinite(q) & np.isfinite(r)
    q = q[finite]
    r = r[finite]

    if sample > 0 and q.size > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(q.size, size=sample, replace=False)
        q = q[idx]
        r = r[idx]

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"QR_map_{time}.png"

    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    ax.scatter(r, q, s=1, alpha=0.25)
    ax.set_xlabel("R")
    ax.set_ylabel("Q")
    ax.set_title(f"Q-R map, time={time}")
    ax.grid(True, linewidth=0.5)
    fig.savefig(out, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return out


def quicklook(params: Params, y: np.ndarray, time: str, outdir: Path, show: bool, dpi: int) -> list[Path]:
    """Generate a small set of useful figures."""
    outputs: list[Path] = []
    j_center = params.Ny // 2
    j_near_wall = min(10, params.Ny - 1)

    outputs.append(plot_profile("ux", params, y, time, outdir, show=False, dpi=dpi))
    outputs.append(plot_slice("ux", params, y, time, "xz", None, j_near_wall, None, outdir, False, "RdBu_r", True, dpi))
    outputs.append(plot_slice("ux", params, y, time, "xz", None, j_center, None, outdir, False, "RdBu_r", True, dpi))
    outputs.append(plot_slice("p_total", params, y, time, "xz", None, j_near_wall, None, outdir, False, "RdBu_r", True, dpi))
    outputs.append(plot_slice("Q", params, y, time, "xz", None, j_near_wall, None, outdir, False, "RdBu_r", True, dpi))
    outputs.append(plot_qr_map(params, time, outdir, show=False, sample=200_000, seed=0, dpi=dpi))

    if show:
        # Show only after all files are created, to avoid blocking batch execution.
        for p in outputs:
            img = plt.imread(p)
            fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(p.name)
            plt.show()
            plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Ret180 channel DNS data.")
    parser.add_argument("--time", default="200", help="Snapshot suffix, e.g. 200")
    parser.add_argument(
        "--field",
        default="ux",
        help="Field name: ux, uy, uz, Q, R, p_rapid, p_slow, p_stokes, p_total",
    )
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xz", help="Slice plane")
    parser.add_argument("--i", type=int, default=None, help="x-index for yz slice")
    parser.add_argument("--j", type=int, default=None, help="y-index for xz slice")
    parser.add_argument("--k", type=int, default=None, help="z-index for xy slice")
    parser.add_argument("--profile", nargs="?", const="ux", help="Plot x-z averaged profile of this field")
    parser.add_argument("--qr-map", action="store_true", help="Plot Q-R scatter map")
    parser.add_argument("--quick", action="store_true", help="Generate a standard quick-look figure set")
    parser.add_argument("--outdir", default="figs", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for slice plots")
    parser.add_argument("--no-symmetric", action="store_true", help="Do not force symmetric color limits")
    parser.add_argument("--sample", type=int, default=200_000, help="Maximum points in Q-R scatter plot")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = read_params(Path("param.txt"))
    y = read_y(Path("yp.dat"), params)
    outdir = Path(args.outdir)

    print(f"Grid: Nx={params.Nx}, Ny={params.Ny}, Nz={params.Nz}")
    print(f"Domain: Lx={params.Lx}, Ly={params.Ly}, Lz={params.Lz}")
    print(f"y range: {y[0]} ... {y[-1]} ({y.size} points)")

    outputs: list[Path] = []
    if args.quick:
        outputs.extend(quicklook(params, y, args.time, outdir, args.show, args.dpi))
    elif args.qr_map:
        outputs.append(plot_qr_map(params, args.time, outdir, args.show, args.sample, seed=0, dpi=args.dpi))
    elif args.profile is not None:
        outputs.append(plot_profile(args.profile, params, y, args.time, outdir, args.show, args.dpi))
    else:
        outputs.append(
            plot_slice(
                args.field,
                params,
                y,
                args.time,
                args.plane,
                args.i,
                args.j,
                args.k,
                outdir,
                args.show,
                args.cmap,
                symmetric=not args.no_symmetric,
                dpi=args.dpi,
            )
        )

    for p in outputs:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
