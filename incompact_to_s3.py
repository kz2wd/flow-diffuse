import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import os
import glob
import re

import numpy as np
import s3fs
import tqdm
import zarr

# Defaults (overridable via CLI flags or env vars).
DEFAULT_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "simulations")
DEFAULT_STORE_PATH = os.environ.get("S3_STORE_PATH", "flow_export/A1")
DEFAULT_SIM_FOLDER = os.environ.get("SIM_FOLDER", "/home/chancrin/chancrin/data/re200-sr05etot")

FIELD_SELECTION = ['ux', 'uy', 'uz', 'pp', 'critq']

def get_snaphot_id(snapshot_path):
    nums = re.findall(r'\d+', snapshot_path)
    return int(nums[-1]) if nums else 100_000_000

def snapshot_list(data_folder):
    matches = glob.glob(f"{data_folder}/snapshot-*.xdmf")
    return sorted(matches, key=get_snaphot_id)

def get_dimensions(snapshot_path):
     root = ET.parse(snapshot_path).getroot()
     topo = root.find('.//Topology')
     dims = topo.attrib['Dimensions']
     return tuple(map(int, dims.split()))

def get_spacing(snapshot_path):
    root = ET.parse(snapshot_path).getroot()
    geo = root.find('.//Geometry')
    data = geo.findall('DataItem')
    dims = ['x', 'y', 'z']
    spacing = {}
    for letter, d in zip(dims, data):
        spacing[letter] = np.array([float(nb) for nb in d.text.strip().split()])
    return spacing

def get_fields(data_folder, snapshot_path, dims, field_selection=None):
    fields = {}
    tree = ET.parse(snapshot_path).getroot()
    for attr in tree.findall(".//Attribute"):
        name = attr.get("Name")
        if field_selection is not None and name not in field_selection:
            continue
        for item in attr.findall("DataItem"):
            file = data_folder / os.path.basename(item.text.strip())
            with open(file, "rb") as f:
                fields[name] = np.frombuffer(f.read(), dtype=np.float64).astype(np.float32).reshape(dims, order='F')
    return fields

def fields_to_array(fields, field_slection):
    arrays = [fields[v] for v in field_slection]
    return np.array(arrays)

def process_snapshot(data_folder, snapshot_path, dims, field_selection, z, i):
    fields = get_fields(data_folder, snapshot_path, dims, field_selection)
    combined = fields_to_array(fields, field_selection)
    z[i] = combined


def export_channel(fs, bucket, store_path, spacing):
    print("    Exporting channel data ...", end=" ")
    map = s3fs.S3Map(root=f"{bucket}/{store_path}/channel.zarr", s3=fs)
    z = zarr.group(
        store=map,
    )
    z.create_dataset(name='x', data=spacing['x'], dtype='f8')
    z.create_dataset(name='y', data=spacing['y'], dtype='f8')
    z.create_dataset(name='z', data=spacing['z'], dtype='f8')
    print("    done.")


def _existing_length(store, bucket, store_path):
    """Length of the time axis if a data.zarr already exists at dest, else 0."""
    try:
        a = zarr.open_array(store=s3fs.S3Map(root=f"{bucket}/{store_path}/data.zarr", s3=store), mode='r')
        return int(a.shape[0])
    except Exception:
        return 0


def export_simulation_data(fs, bucket, store_path, field_selection, data_folder, snapshots, dims, offset=0, existing_len=0):
    print("    Exporting simulation data ...")
    map = s3fs.S3Map(root=f"{bucket}/{store_path}/data.zarr", s3=fs)
    total_len = existing_len + offset + len(snapshots)
    z = zarr.open(
        store=map,
        mode="a" if (existing_len or offset) else "w",
        shape=(total_len, len(field_selection), *dims),
        chunks=(1, len(field_selection), *dims),
        dtype='f4',
    )

    for i, snapshot in enumerate(tqdm.tqdm(snapshots, desc="Processing snapshots")):
        process_snapshot(data_folder, snapshot, dims, field_selection, z, offset + i)

    print("    done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a local Incompact3d simulation (snapshot-*.xdmf) to an S3-compatible store (MinIO, SeaweedFS, ...) as zarr."
    )
    parser.add_argument("--sim-folder", default=DEFAULT_SIM_FOLDER,
                        help="Path to the Incompact3d simulation folder (contains a 'data' subfolder with snapshot-*.xdmf).")
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT,
                        help="S3 endpoint URL (default: %(default)s).")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET,
                        help="Target S3 bucket (default: %(default)s).")
    parser.add_argument("--store-path", default=DEFAULT_STORE_PATH,
                        help="Destination path inside the bucket (default: %(default)s).")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip the interactive 'press ENTER' confirmation.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Index to start appending at (continuation of an existing dataset). "
                             "E.g. offset=1650 appends after the current 0-1649. Default: 0.")
    return parser.parse_args()


def main():
    args = parse_args()

    data_folder = Path(args.sim_folder) / "data"
    field_selection = FIELD_SELECTION

    fs = s3fs.S3FileSystem(profile='default', client_kwargs={
        'endpoint_url': args.endpoint_url
    })

    snapshots = snapshot_list(data_folder)
    dims = get_dimensions(snapshots[0])
    spacing = get_spacing(snapshots[0])
    print(f"  Dimensions: {dims}")
    print(f"  Fields: {field_selection}")
    print(f"  Total snapshots: {len(snapshots)}")
    print(f"  Target s3://{args.bucket}/{args.store_path}")
    if not args.no_confirm:
        input("  Press [ENTER] to start export")
    print("  Starting export...")

    export_channel(fs, args.bucket, args.store_path, spacing)
    existing_len = _existing_length(fs, args.bucket, args.store_path)
    if args.offset:
        print(f"  Existing length: {existing_len}  |  offset: {args.offset}  |  appending {len(snapshots)}")
    export_simulation_data(fs, args.bucket, args.store_path, field_selection, data_folder, snapshots, dims,
                           offset=args.offset, existing_len=existing_len)
    print(f"  Export complete! Data stored at s3://{args.bucket}/{args.store_path}")


if __name__ == "__main__":
    main()
