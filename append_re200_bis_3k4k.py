"""Append NEW local Incompact3d snapshots to the remote A1 data.zarr.

Auto-detects the "new portion":
  - reads the remote data.zarr length L (remote holds indices 0..L-1)
  - reads all local snapshot indices
  - new portion = local indices in [L, max_local]   (i.e. everything the remote lacks)
So re-running this script later just appends whatever new snapshots appeared --
you never hardcode a range.

Verified remote state (A1/data.zarr, zarr v2):
  shape (3001, 5, 128, 129, 128), chunks (1,5,128,129,128), dtype f4, blosc/lz4
  local snapshot N is stored at remote index N (local 3000 == remote 3000, diff 0.0)

channel.zarr is left untouched (already correct).

Safety:
  - only writes to indices >= L (never touches 0..L-1)
  - resumable: if it dies mid-run, re-running recomputes new portion =
    [L, max_local] and overwrites those indices (idempotent)
  - refuses to run if local indices are not contiguous from L (would leave gaps)
"""
from __future__ import annotations
import sys
import re
import glob
from pathlib import Path

import numpy as np
import s3fs
import zarr
import tqdm

# ---- reuse the project's local-reader helpers (pure functions) ----
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i2s", "/data3/chancrin/code/flow-diffuse/incompact_to_s3.py")
i2s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i2s)

# ---- config ----
ENDPOINT = "http://localhost:8333"
BUCKET = "simulations"
STORE = "flow_export/A1"
LOCAL_SIM = "/data2/chancrin/data/re200-sr05etot-bis"
EXPECTED_SPATIAL = (5, 128, 129, 128)
FIELD_SELECTION = i2s.FIELD_SELECTION   # ['ux','uy','uz','pp','critq']


def _snap_num(path) -> int:
    m = re.findall(r"\d+", str(path))
    return int(m[-1]) if m else 0


def main():
    fs = s3fs.S3FileSystem(profile='default',
                           client_kwargs={'endpoint_url': ENDPOINT})
    store = s3fs.S3Map(root=f"{BUCKET}/{STORE}/data.zarr", s3=fs, check=False)
    a = zarr.open_array(store=store, mode='r')
    cur_len = a.shape[0]
    assert a.shape[1:] == EXPECTED_SPATIAL, \
        f"unexpected spatial shape {a.shape[1:]} (expected {EXPECTED_SPATIAL})"
    print(f"remote data.zarr: shape {a.shape}  (holds indices 0..{cur_len-1})")

    # ---- local snapshot indices ----
    data_folder = Path(LOCAL_SIM) / "data"
    local_snaps = sorted(glob.glob(str(data_folder / "snapshot-*.xdmf")),
                         key=_snap_num)
    local_idxs = [_snap_num(p) for p in local_snaps]
    local_min, local_max = min(local_idxs), max(local_idxs)
    print(f"local snapshots: {len(local_idxs)}  (min={local_min}, max={local_max})")

    # ---- new portion = local indices >= cur_len ----
    new_idxs = [i for i in local_idxs if i >= cur_len]
    new_idxs.sort()
    if not new_idxs:
        print("Nothing to do: remote already has all local snapshots.")
        return

    # contiguity check: new portion must start exactly at cur_len and be dense,
    # otherwise we'd write gaps / skip indices.
    if new_idxs[0] != cur_len:
        raise SystemExit(
            f"NEW: local indices jump from {cur_len-1} to {new_idxs[0]} "
            f"(gap of {new_idxs[0]-cur_len}). Refusing to leave gaps. "
            f"Confirm the missing indices are intentional before proceeding.")
    gaps = [b - a for a, b in zip(new_idxs, new_idxs[1:]) if b - a != 1]
    if gaps:
        raise SystemExit(f"NEW: non-contiguous new indices (gaps): {gaps[:20]}...")

    print(f"NEW portion: {len(new_idxs)} snapshots  "
          f"({new_idxs[0]}..{new_idxs[-1]})  -> remote {cur_len}..{cur_len+len(new_idxs)-1}")
    print(f"final shape will be ({cur_len + len(new_idxs)}, {EXPECTED_SPATIAL[0]}, ...)\n")

    # ---- resize (zarr v2 metadata) ----
    a = zarr.open_array(store=store, mode='r+')
    new_len = cur_len + len(new_idxs)
    a.resize((new_len, *EXPECTED_SPATIAL))
    print(f"resized {cur_len} -> {new_len}\n")

    # ---- append ----
    dims = i2s.get_dimensions(str(local_snaps[0]))
    for local_idx in tqdm.tqdm(new_idxs, desc="appending"):
        snap_path = str(data_folder / f"snapshot-{local_idx}.xdmf")
        fields = i2s.get_fields(data_folder, snap_path, dims, FIELD_SELECTION)
        arr = i2s.fields_to_array(fields, FIELD_SELECTION)
        remote_idx = cur_len + (local_idx - new_idxs[0])
        a[remote_idx] = arr

    # ---- verify ----
    a = zarr.open_array(store=store, mode='r')
    print("\n=== VERIFY ===")
    print("final shape:", a.shape)
    for i in [cur_len - 1, cur_len, new_len - 1]:
        v = np.asarray(a[i])
        print(f"  idx {i}: finite={np.isfinite(v).sum()}/{v.size}  "
              f"max|v|={np.abs(v).max():.4f}")
    print(f"\nDone. s3://{BUCKET}/{STORE}/data.zarr  shape {a.shape}")


if __name__ == "__main__":
    main()
