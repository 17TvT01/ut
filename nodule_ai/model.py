from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = True) -> None:
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_depth = x2.size(2) - x1.size(2)
        diff_height = x2.size(3) - x1.size(3)
        diff_width = x2.size(4) - x1.size(4)
        x1 = nn.functional.pad(
            x1,
            [
                diff_width // 2,
                diff_width - diff_width // 2,
                diff_height // 2,
                diff_height - diff_height // 2,
                diff_depth // 2,
                diff_depth - diff_depth // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )
        self.se = SqueezeExcitation3D(out_channels)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.block(x)
        out = self.dropout(out)
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class ComplexDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.block = ResidualConvBlock3D(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class ComplexUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        mode: str = "trilinear",
    ) -> None:
        super().__init__()
        if mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.block = ResidualConvBlock3D(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_depth = skip.size(2) - x.size(2)
        diff_height = skip.size(3) - x.size(3)
        diff_width = skip.size(4) - x.size(4)
        if diff_depth != 0 or diff_height != 0 or diff_width != 0:
            x = F.pad(
                x,
                [
                    diff_width // 2,
                    diff_width - diff_width // 2,
                    diff_height // 2,
                    diff_height - diff_height // 2,
                    diff_depth // 2,
                    diff_depth - diff_depth // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ComplexUNet3D(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        base_filters: int = 32,
        dropout: float = 0.1,
        upsample_mode: str = "trilinear",
    ) -> None:
        super().__init__()
        self.stem = ResidualConvBlock3D(n_channels, base_filters, dropout=dropout)
        self.down1 = ComplexDown3D(base_filters, base_filters * 2, dropout=dropout)
        self.down2 = ComplexDown3D(base_filters * 2, base_filters * 4, dropout=dropout)
        self.down3 = ComplexDown3D(base_filters * 4, base_filters * 8, dropout=dropout)
        self.bottleneck = ResidualConvBlock3D(base_filters * 8, base_filters * 16, dropout=dropout)
        self.up1 = ComplexUp3D(
            base_filters * 16,
            base_filters * 8,
            base_filters * 8,
            dropout=dropout,
            mode=upsample_mode,
        )
        self.up2 = ComplexUp3D(
            base_filters * 8,
            base_filters * 4,
            base_filters * 4,
            dropout=dropout,
            mode=upsample_mode,
        )
        self.up3 = ComplexUp3D(
            base_filters * 4,
            base_filters * 2,
            base_filters * 2,
            dropout=dropout,
            mode=upsample_mode,
        )
        self.up4 = ComplexUp3D(
            base_filters * 2,
            base_filters,
            base_filters,
            dropout=dropout,
            mode=upsample_mode,
        )
        self.out = nn.Conv3d(base_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class UNet3D(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_filters: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv3D(n_channels, base_filters)
        self.down1 = Down3D(base_filters, base_filters * 2)
        self.down2 = Down3D(base_filters * 2, base_filters * 4)
        self.down3 = Down3D(base_filters * 4, base_filters * 8)
        self.down4 = Down3D(base_filters * 8, base_filters * 8)
        self.up1 = Up3D(base_filters * 16, base_filters * 4)
        self.up2 = Up3D(base_filters * 8, base_filters * 2)
        self.up3 = Up3D(base_filters * 4, base_filters)
        self.up4 = Up3D(base_filters * 2, base_filters)
        self.outc = OutConv3D(base_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


__all__ = [
    "DoubleConv3D",
    "Down3D",
    "Up3D",
    "OutConv3D",
    "UNet3D",
    "SqueezeExcitation3D",
    "ResidualConvBlock3D",
    "ComplexDown3D",
    "ComplexUp3D",
    "ComplexUNet3D",
]
