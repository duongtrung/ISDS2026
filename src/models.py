"""Shared architecture: encoder over appearance+motion channels, energy head,
amortized init net. Fully convolutional -> resolution-agnostic. The width `ch` is
configurable (default 32 = the original small prototype; larger for the capacity
scenario). ch=32 is bit-identical to the original models, so old checkpoints load."""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_ch=4, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class EnergyNet(nn.Module):
    """E_theta(mask | features) -> scalar per sample. Low energy = plausible mask."""

    def __init__(self, feat_ch=32, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch + 1, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, feat, mask):
        e_map = self.net(torch.cat([feat, mask], dim=1))
        return e_map.mean(dim=(1, 2, 3)) * 10.0


class InitNet(nn.Module):
    """Amortized sampler: one-shot mask-logit proposal from features."""

    def __init__(self, feat_ch=32, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, feat):
        return self.net(feat)


# --- input modality helpers (RGB = channels 0:3, flow = channels 3:5) ---
MODALITY_CH = {'both': 5, 'rgb': 3, 'flow': 2}


def slice_modality(x, modality):
    if modality == 'rgb':
        return x[:, :3]
    if modality == 'flow':
        return x[:, 3:5]
    return x
