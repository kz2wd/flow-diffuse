"""Copy a zarr store (dataset + channel config) between two S3-compatible stores
(e.g. old MinIO -> new SeaweedFS), chunk by chunk, without re-encoding.

Usage (defaults below assume old MinIO on :9000, new SeaweedFS on :8333):

    uv run python s3_to_s3.py
    uv run python s3_to_s3.py --source-endpoint http://localhost:9000 \
        --dest-endpoint http://localhost:8333 \
        --source-bucket simulations --dest-bucket simulations \
        --store-path flow_export/A1 --items data.zarr channel.zarr
"""

import argparse
import os

import numpy as np
import s3fs
import tqdm
import zarr

DEFAULT_SOURCE_ENDPOINT = os.environ.get("S3_SOURCE_ENDPOINT", "http://localhost:9000")
DEFAULT_DEST_ENDPOINT = os.environ.get("S3_DEST_ENDPOINT", "http://localhost:8333")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "simulations")
DEFAULT_STORE_PATH = os.environ.get("S3_STORE_PATH", "flow_export/A1")
DEFAULT_ITEMS = "data.zarr,channel.zarr"


def make_fs(endpoint_url):
    return s3fs.S3FileSystem(profile='default', client_kwargs={
        'endpoint_url': endpoint_url,
    })


def copy_group(source_zarr, dest_root, name, dest_fs):
    """Copy a single zarr array (dataset or channel config) source -> dest,
    chunk by chunk, so neither side holds the full array in memory."""
    src = source_zarr[name]
    dest_map = s3fs.S3Map(root=f"{dest_root}/{name}", s3=dest_fs)
    z = zarr.open(
        store=dest_map,
        mode="w",
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
    )

    # Enumerate chunk indices (one per output chunk) without materializing the array.
    n_chunks = [max(1, s // c) for s, c in zip(src.shape, src.chunks)]
    idxs = np.indices(n_chunks).reshape(len(n_chunks), -1).T
    for chunk_idx in tqdm.tqdm(idxs, total=len(idxs), desc=name):
        sel = tuple(slice(a * c, a * c + c) for a, c in zip(chunk_idx, src.chunks))
        z[sel] = src[sel]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a zarr dataset + channel config between two S3-compatible stores "
                    "(e.g. old MinIO -> new SeaweedFS), chunk by chunk."
    )
    parser.add_argument("--source-endpoint", default=DEFAULT_SOURCE_ENDPOINT,
                        help="Source S3 endpoint URL (old MinIO). Default: %(default)s.")
    parser.add_argument("--dest-endpoint", default=DEFAULT_DEST_ENDPOINT,
                        help="Dest S3 endpoint URL (new SeaweedFS). Default: %(default)s.")
    parser.add_argument("--source-bucket", default=DEFAULT_BUCKET,
                        help="Source S3 bucket. Default: %(default)s.")
    parser.add_argument("--dest-bucket", default=DEFAULT_BUCKET,
                        help="Dest S3 bucket. Default: %(default)s.")
    parser.add_argument("--store-path", default=DEFAULT_STORE_PATH,
                        help="Path inside the bucket holding the zarr store. Default: %(default)s.")
    parser.add_argument("--items", default=DEFAULT_ITEMS,
                        help="Comma-separated zarr items to copy (default: %(default)s).")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip the interactive 'press ENTER' confirmation.")
    return parser.parse_args()


def main():
    args = parse_args()

    source_fs = make_fs(args.source_endpoint)
    dest_fs = make_fs(args.dest_endpoint)

    source_root = f"{args.source_bucket}/{args.store_path}"
    dest_root = f"{args.dest_bucket}/{args.store_path}"
    items = [i.strip() for i in args.items.split(",") if i.strip()]

    source_map = s3fs.S3Map(root=f"{args.source_bucket}/{args.store_path}", s3=source_fs)
    source_zarr = zarr.open(store=source_map, mode="r")

    print(f"  Source: s3://{args.source_bucket}/{args.store_path}  (endpoint {args.source_endpoint})")
    print(f"  Dest:   s3://{args.dest_bucket}/{args.store_path}  (endpoint {args.dest_endpoint})")
    print(f"  Items:  {items}")
    if not args.no_confirm:
        input("  Press [ENTER] to start copy")

    for name in items:
        if name not in source_zarr:
            print(f"  [skip] {name!r} not found in source store")
            continue
        copy_group(source_zarr, dest_root, name, dest_fs)

    print(f"  Copy complete! Data now at s3://{args.dest_bucket}/{args.store_path}")


if __name__ == "__main__":
    main()
