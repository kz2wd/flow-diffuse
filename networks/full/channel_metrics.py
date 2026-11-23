"""Validation diagnostics for normalized [B, C, X, Y, Z] channel fields."""
from __future__ import annotations

import torch


class SpectralValidation:
    """Accumulate additive MSE contributions in radial XZ Fourier bands.

    Radii use integer mode numbers, not dimensional wavenumbers. Keep DC
    separate and group the remaining modes into [1, 2), [2, 4), etc.
    Equal batch weighting matches the validation objective, including a short last batch.
    """

    def __init__(self, *, image_samples: int = 0):
        self.image_samples = image_samples
        self.slices: list[torch.Tensor] = []
        self.y_index = 0
        self._batch_count = 0
        self._band_energy: torch.Tensor | None = None

    def _init_bands(self, nx: int, nz: int, device: torch.device):
        kx = torch.fft.fftfreq(nx, d=1 / nx).round()
        kz = torch.arange(nz // 2 + 1)
        radius_sq = kx[:, None].square() + kz.square()
        max_radius_sq = int(radius_sq.max())
        edges = []
        edge = 1
        while edge * edge <= max_radius_sq:
            edges.append(edge)
            edge *= 2
        bands = torch.bucketize(radius_sq.flatten(), torch.tensor(edges).square(), right=True)
        names = ["dc", *(f"k_{edge}_to_{2 * edge}" for edge in edges)]
        self._names = [(i, name) for i, name in enumerate(names) if (bands == i).any()]
        self._bands = bands.to(device)
        self._band_energy = torch.zeros(len(names), device=device, dtype=torch.float64)

        # The omitted negative kz modes have the same total energy in each band.
        weights = torch.full((nz // 2 + 1,), 2.0, device=device)
        weights[0] = 1.0
        if nz % 2 == 0:
            weights[-1] = 1.0
        self._weights = weights

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        b, c, nx, ny, nz = target.shape
        if self._band_energy is None:
            self._init_bands(nx, nz, target.device)
        # Chunk y to bound the FFT workspace without dropping any planes.
        mode_energy = torch.zeros((nx, nz // 2 + 1), device=target.device, dtype=torch.float64)
        for start in range(0, ny, 16):
            planes = slice(start, start + 16)
            error = prediction[..., planes, :] - target[..., planes, :]
            spectrum = torch.fft.rfft2(error, dim=(-3, -1), norm="ortho")
            mode_energy += spectrum.abs().square().sum(dim=(0, 1, 3), dtype=torch.float64)
        mode_mse = (mode_energy * self._weights) / (b * c * nx * ny * nz)
        self._band_energy.scatter_add_(0, self._bands, mode_mse.flatten())
        self._batch_count += 1

        remaining = self.image_samples - len(self.slices)
        if remaining > 0:
            self.y_index = ny // 2
            fields = torch.stack((prediction[:remaining, :, :, self.y_index, :],
                                  target[:remaining, :, :, self.y_index, :]), dim=1)
            fft = torch.fft.fft2(fields, norm="ortho")
            power = torch.fft.fftshift(fft.abs().square(), dim=(-2, -1))
            self.slices.extend(power.cpu().unbind())

    def scalars(self) -> dict[str, float]:
        if self._band_energy is None:
            return {}
        mse = (self._band_energy / self._batch_count).tolist()
        return {f"val/spectral_mse/{name}": mse[i] for i, name in self._names}


def render_spectral_slice(power: torch.Tensor, channel_names: tuple[str, ...]):
    """Render paired, FFT-shifted XZ power spectra, with one row per channel."""
    from PIL import Image, ImageDraw

    _, channels, nx, nz = power.shape
    width, height = max(184, nx), max(100, nz)
    gap, header, footer = 8, 18, 18
    row_height = header + height + footer
    image = Image.new("RGB", (2 * width + gap, channels * row_height), "white")
    draw = ImageDraw.Draw(image)
    for c in range(channels):
        row = c * row_height
        # Use the same six-decade range for prediction and target in this row.
        log_power = power[:, c].clamp_min(1e-30).log10()
        high = max(log_power.max().item(), -24.0)
        low = high - 6.0
        gray = ((log_power - low) / (high - low)).clamp(0, 1).mul(255).byte()
        for j, name in enumerate(("Reconstruction", "Ground truth")):
            x = j * (width + gap)
            draw.text((x + 4, row + 2), f"{name}: {channel_names[c]}", fill="black")
            # kx increases to the right, kz upwards; DC is at the center.
            panel = Image.fromarray(gray[j].T.flip(0).contiguous().numpy())
            image.paste(panel.resize((width, height), Image.Resampling.NEAREST), (x, row + header))
        draw.text((4, row + header + height + 2),
                  f"log10 power: {low:.1f} to {high:.1f}   |   horizontal: kx, vertical: kz", fill="black")
    return image
