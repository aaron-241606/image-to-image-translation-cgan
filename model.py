"""
Task-04: pix2pix cGAN — Generator (U-Net) + Discriminator (PatchGAN)
Prodigy Infotech Internship
"""

import torch
import torch.nn as nn


# ─── U-Net Generator ────────────────────────────────────────────────────────

class UNetBlock(nn.Module):
    """Single encoder or decoder block for the U-Net generator."""

    def __init__(self, in_channels, out_channels, down=True, use_bn=True, dropout=False, activation="relu"):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout:
            layers.append(nn.Dropout(0.5))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net based Generator for pix2pix.
    Input:  (B, 3, 256, 256) — source image
    Output: (B, 3, 256, 256) — translated image
    """

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2))   # 128
        self.enc2 = UNetBlock(features,     features * 2,  down=True,  use_bn=True,  activation="leaky")  # 64
        self.enc3 = UNetBlock(features * 2, features * 4,  down=True,  use_bn=True,  activation="leaky")  # 32
        self.enc4 = UNetBlock(features * 4, features * 8,  down=True,  use_bn=True,  activation="leaky")  # 16
        self.enc5 = UNetBlock(features * 8, features * 8,  down=True,  use_bn=True,  activation="leaky")  # 8
        self.enc6 = UNetBlock(features * 8, features * 8,  down=True,  use_bn=True,  activation="leaky")  # 4
        self.enc7 = UNetBlock(features * 8, features * 8,  down=True,  use_bn=True,  activation="leaky")  # 2
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())        # 1

        # Decoder (upsampling with skip connections)
        self.dec1 = UNetBlock(features * 8,      features * 8, down=False, use_bn=True, dropout=True,  activation="relu")
        self.dec2 = UNetBlock(features * 8 * 2,  features * 8, down=False, use_bn=True, dropout=True,  activation="relu")
        self.dec3 = UNetBlock(features * 8 * 2,  features * 8, down=False, use_bn=True, dropout=True,  activation="relu")
        self.dec4 = UNetBlock(features * 8 * 2,  features * 8, down=False, use_bn=True, dropout=False, activation="relu")
        self.dec5 = UNetBlock(features * 8 * 2,  features * 4, down=False, use_bn=True, dropout=False, activation="relu")
        self.dec6 = UNetBlock(features * 4 * 2,  features * 2, down=False, use_bn=True, dropout=False, activation="relu")
        self.dec7 = UNetBlock(features * 2 * 2,  features,     down=False, use_bn=True, dropout=False, activation="relu")
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        bn = self.bottleneck(e7)

        d1 = self.dec1(bn)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        return self.final(torch.cat([d7, e1], dim=1))


# ─── PatchGAN Discriminator ──────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator.
    Input:  concatenated (source, target) images → (B, 6, 256, 256)
    Output: patch predictions → (B, 1, 30, 30)
    """

    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features,     features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, 4, 1, 1),
        )

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        return self.model(x)


def initialize_weights(model):
    """Initialize weights with N(0, 0.02) as in the original pix2pix paper."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
