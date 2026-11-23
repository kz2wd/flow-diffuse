"""DataNet architecture and factory for Channel-flow reconstruction."""
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding="same"),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding="same"),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class DataNet(nn.Module):
    def __init__(self, c: int = 64, n_blocks: int = 3, dy: int = 129, out_ch: int = 3):
        super().__init__()
        self.dy = dy
        self.c = c
        self.core = nn.Sequential(
            nn.Conv3d(1, c, 3, padding="same"),
            *[ConvBlock3D(c) for _ in range(n_blocks)],
            nn.Conv3d(c, out_ch, 3, padding="same"),
        )

    def forward(self, p):
        x = p.expand(-1, -1, -1, self.dy, -1)
        return self.core(x)


def make_model(arch: dict, device: str) -> DataNet:
    return DataNet(
        c=int(arch.get("c", 64)),
        n_blocks=int(arch.get("n_blocks", 3)),
        dy=int(arch.get("dy", 129)),
        out_ch=int(arch.get("out_ch", 3)),
    ).to(device)
