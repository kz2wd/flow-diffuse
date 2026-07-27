import torch
import numpy as np
import s3fs
import zarr

from diffusers import UNet2DModel
from diffusers import DDPMScheduler

import torch.optim as optim
import torch.nn as nn

model = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

def train_step(model, noise_scheduler, clean_image, optimizer, loss):
    batch_size = clean_image.shape[0]
    optimizer.zero_grad()
    noise = torch.randn_like(clean_image)
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,))
    image_t = noise_scheduler.add_noise(clean_image, noise, t)
    noise_pred = model(image_t, t, return_dict=False)[0]
    l = loss(noise_pred, noise)
    l.backward()
    optimizer.step()
    return l.item()


def diffuse(model, noise_scheduler, prediction_shape):
    sample = torch.randn(prediction_shape)
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        noise_pred = model(sample, t, return_dict=False)[0]
        sample = noise_scheduler.step(noise_pred, t, sample, return_dict=False)[0]
    return sample


class S3FlowDataset(torch.utils.data.Dataset):
    def __init__(self, y_index):
        self.y_index = y_index
        self._fs = None
        self._s3_map = None

    def _ensure_connection(self):
        if self._fs is None:
            file_path = "simulations/re200-sr05etot.zarr"
            self._fs = s3fs.S3FileSystem(profile='default', client_kwargs={
                'endpoint_url': 'http://localhost:9000',
            })
            s3_map = s3fs.S3Map(root=file_path, s3=self.fs, check=False)
            self._s3_map = zarr.open(store=s3_map, mode="r")

    @property
    def z(self):
        self._ensure_connection()
        return self._s3_map

    def __len__(self):
        return self.z["Y"].shape[0]
    def __getitem__(self, i):
        return self.z["Y"][i, :, :, self.y_index, :]


dataset = S3FlowDataset(0)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True,
    num_workers=2, pin_memory=True,
)

loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 1
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        epoch_loss += train_step(model, noise_scheduler, batch, optimizer, loss_fn)

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss* {epoch_loss / len(dataloader):.4f}")
