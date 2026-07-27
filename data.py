import numpy as np
import s3fs
import zarr

file_path = "simulations/re200-sr05etot.zarr"
fs = s3fs.S3FileSystem(profile='default', client_kwargs={
    'endpoint_url': 'http://localhost:9000'
})
map = s3fs.S3Map(root=file_path, s3=fs, check=False)

# %%

z = zarr.open(store=map, mode="r")

print(z['Y'].shape)
