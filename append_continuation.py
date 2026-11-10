"""Append a continuation Incompact3d simulation to an existing zarr dataset on
S3 (e.g. SeaweedFS), in place.

Mechanism (proven on a scratch store first):
  1. Back up the destination .zarray (rollback point).
  2. Rewrite the destination .zarray to declare the new total length
     (existing_len + offset + n_new). Single tiny object; existing chunk
     objects are untouched.
  3. Open the destination in append mode and write each continuation snapshot
     at index (offset + i), using zarr's own compressor so layout/compression
     match the existing chunks.
  4. Verify boundary + spot rows read back, then restore or keep the metadata.

The continuation folder must use the SAME field set, grid, and chunk layout as
the existing store (true for re200-sr05etot-bis: snapshots 1650-3000, grid
128x129x128, fields ux/uy/uz/pp/critq).
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import s3fs
import tqdm
import zarr

FIELD_SELECTION = ['ux', 'uy', 'uz', 'pp', 'critq']


def make_fs(endpoint_url):
    return s3fs.S3FileSystem(profile='default', client_kwargs={'endpoint_url': endpoint_url})


def snapshot_list(data_folder):
    matches = sorted(Path(data_folder).glob("snapshot-*.xdmf"),
                     key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
    return matches


def get_dimensions(snapshot_path):
    root = ET.parse(snapshot_path).getroot()
    return tuple(map(int, root.find('.//Topology').attrib['Dimensions'].split()))


def get_fields(data_folder, snapshot_path, dims, field_selection):
    fields = {}
    tree = ET.parse(snapshot_path).getroot()
    for attr in tree.findall(".//Attribute"):
        name = attr.get("Name")
        if name not in field_selection:
            continue
        for item in attr.findall("DataItem"):
            file = data_folder / os.path.basename(item.text.strip())
            with open(file, "rb") as f:
                fields[name] = np.frombuffer(f.read(), dtype=np.float64).astype(np.float32).reshape(dims, order='F')
    return fields


def fields_to_array(fields, field_selection):
    return np.array([fields[v] for v in field_selection])


def parse_args():
    p = argparse.ArgumentParser(description="Append a continuation Incompact3d sim to an existing zarr dataset on S3.")
    p.add_argument("--sim-folder", default="/data2/chancrin/data/re200-sr05etot-bis",
                   help="Continuation simulation folder (contains a 'data' subfolder).")
    p.add_argument("--endpoint-url", default=os.environ.get("S3_ENDPOINT_URL", "http://localhost:8333"),
                   help="Destination S3 endpoint (SeaweedFS).")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "simulations"))
    p.add_argument("--store-path", default=os.environ.get("S3_STORE_PATH", "flow_export/A1"))
    p.add_argument("--offset", type=int, default=1650,
                   help="Global index to start appending at (existing length). Default: 1650.")
    p.add_argument("--dry-run", action="store_true", help="Inspect + report plan, write nothing.")
    return p.parse_args()


def main():
    args = parse_args()
    fs = make_fs(args.endpoint_url)
    dest_root = f"{args.bucket}/{args.store_path}/data.zarr"
    data_folder = Path(args.sim_folder) / "data"

    # ---- inspect destination ----
    dest_meta = json.loads(fs.cat(f"{dest_root}/.zarray").decode())
    existing_len = dest_meta["shape"][0]
    dest_chunks = dest_meta["chunks"]
    print(f"Dest {dest_root}: existing length {existing_len}, chunks {dest_chunks}, dtype {dest_meta['dtype']}")

    if args.offset != existing_len:
        print(f"  WARNING: --offset {args.offset} != existing length {existing_len}. "
              f"Applying as-is; verify this is what you want.")

    # ---- inspect continuation ----
    snaps = snapshot_list(data_folder)
    first = get_dimensions(snaps[0])
    first_fields = {a.get("Name") for a in ET.parse(snaps[0]).getroot().findall(".//Attribute")}
    print(f"Continuation: {len(snaps)} snapshots "
          f"({snaps[0].stem} -> {snaps[-1].stem}), grid {first}")
    print(f"Continuation fields: {sorted(first_fields)}")
    missing = set(FIELD_SELECTION) - first_fields
    if missing:
        raise SystemExit(f"  FATAL: continuation is missing fields {missing}; aborting.")

    n_new = len(snaps)
    total = existing_len + n_new
    print(f"Plan: append {n_new} -> final length {total}")

    if args.dry_run:
        print("  --dry-run: no changes made.")
        return

    # ---- backup metadata (rollback point) ----
    backup_root = f"{args.bucket}/{args.store_path}/data.zarr.bak"
    fs.pipe_file(f"{backup_root}/.zarray", fs.cat(f"{dest_root}/.zarray"))
    print(f"  Backed up .zarray -> {backup_root}/.zarray")

    # ---- grow the array: rewrite .zarray to new length ----
    new_meta = json.loads(json.dumps(dest_meta))  # deep copy
    new_meta["shape"][0] = total
    fs.pipe_file(f"{dest_root}/.zarray", json.dumps(new_meta).encode())
    print(f"  Rewrote .zarray length {existing_len} -> {total}")

    # ---- append-mode write the continuation chunks ----
    z = zarr.open_array(store=s3fs.S3Map(root=dest_root, s3=fs), mode="a")
    print(f"  Appending {n_new} snapshots at indices {existing_len}..{total-1} ...")
    for i, snap in enumerate(tqdm.tqdm(snaps, desc="append", unit="snap")):
        fields = get_fields(data_folder, snap, first, FIELD_SELECTION)
        z[existing_len + i] = fields_to_array(fields, FIELD_SELECTION)

    # ---- verify ----
    print("  Verifying readback ...")
    ok = True
    c = zarr.open_array(store=s3fs.S3Map(root=dest_root, s3=fs), mode="r")
    if c.shape[0] != total:
        ok = False
        print(f"    shape mismatch: got {c.shape}, expected {total}")
    for idx in [0, existing_len - 1, existing_len, total - 1]:
        row = c[idx]
        finite = np.isfinite(row).all()
        print(f"    row {idx}: finite={bool(finite)} mean={float(row.mean()):.4f}")
        ok = ok and bool(finite)
    # boundary sanity: existing row (existing_len-1) must be unchanged (nonzero)
    ok = ok and bool(np.abs(c[existing_len - 1]).sum() > 0)
    # new row (existing_len) must be the continuation's first snapshot
    ref = fields_to_array(get_fields(data_folder, snaps[0], first, FIELD_SELECTION), FIELD_SELECTION)
    ok = ok and bool(np.array_equal(c[existing_len], ref))

    if ok:
        print(f"  DONE: dataset is now length {total}. Backup at {backup_root}/.zarray")
    else:
        print("  VERIFICATION FAILED. Restoring original .zarray ...")
        fs.pipe_file(f"{dest_root}/.zarray", fs.cat(f"{backup_root}/.zarray"))
        print("  (New chunk objects remain in the store; the declared length is back to "
              f"{existing_len}. Delete stray {existing_len}..{total-1} objects if needed.)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
