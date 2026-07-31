import xml.etree.ElementTree as ET
from pathlib import Path
import zarr
import s3fs
import numpy as np
import os
import glob
import re
import tqdm

S3_BUCKET = "simulations"
STORE_PATH = "flow_export/A1"
FIELD_SELECTION = ['ux', 'uy', 'uz', 'pp', 'critq']
SIM_FILDER = Path("/home/chancrin/chancrin/data/re200-sr05etot")
DATA_FOLDER = SIM_FILDER / "data"

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

def get_fields(data_folder, snapshot_path, dims):
    fields = {}
    tree = ET.parse(snapshot_path).getroot()
    for attr in tree.findall(".//Attribute"):
        for item in attr.findall("DataItem"):
            file = data_folder / os.path.basename(item.text.strip())
            with open(file, "rb") as f:
                fields[attr.get("Name")] = np.frombuffer(f.read(), dtype=np.float64).astype(np.float32).reshape(dims, order='F')
    return fields

def fields_to_array(fields, field_slection):
    arrays = [fields[v] for v in field_slection]
    return np.array(arrays)

def process_snapshot(data_folder, snapshot_path, dims, field_selection, z, i):
    fields = get_fields(data_folder, snapshot_path, dims)
    combined = fields_to_array(fields, field_selection)
    z[i] = combined


def export_channel(fs, spacing):
    print("    Exporting channel data ...", end=" ")
    map = s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/channel.zarr", s3=fs)
    z = zarr.group(
        store=map,
    )
    z.create_dataset(name='x', data=spacing['x'], dtype='f8')
    z.create_dataset(name='y', data=spacing['y'], dtype='f8')
    z.create_dataset(name='z', data=spacing['z'], dtype='f8')
    print("    done.")

def export_simulation_data(fs, num_fields, snapshots, dims):
    print("    Exporting simulation data ...")
    map = s3fs.S3Map(root=f"{S3_BUCKET}/{STORE_PATH}/data.zarr", s3=fs)
    z = zarr.open(
        store=map,
        mode="w",
        shape=(len(snapshots), num_fields, *dims),
        chunks=(1, num_fields, *dims),
        dtype='f4',
    )

    for i, snapshot in enumerate(tqdm.tqdm(snapshots, desc="Processing snapshots")):
        process_snapshot(DATA_FOLDER, snapshot, dims, FIELD_SELECTION, z, i)

    print("    done.")


def main():

    fs = s3fs.S3FileSystem(profile='default', client_kwargs={
        'endpoint_url': 'http://localhost:9000'
    })

    num_fields = len(FIELD_SELECTION)
    snapshots = snapshot_list(DATA_FOLDER)
    dims = get_dimensions(snapshots[0])
    spacing = get_spacing(snapshots[0])
    print(f"  Dimensions: {dims}")
    print(f"  Fields: {FIELD_SELECTION}")
    print(f"  Total snapshots: {len(snapshots)}")
    print(f"  Target s3://{S3_BUCKET}/{STORE_PATH}")
    input("  Press [ENTER] to start export")
    print("  Starting export...")

    export_channel(fs, spacing)
    export_simulation_data(fs, num_fields, snapshots, dims)
    print(f"  Export complete! Data stored at s3://{S3_BUCKET}/{STORE_PATH}")

if __name__ == "__main__":
    main()
