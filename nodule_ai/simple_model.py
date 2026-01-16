#!/usr/bin/env python
"""
Simplified UNet3D - Lighter model for small dataset
Parameters: ~500K (vs ~3M in ComplexUNet3D)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleUNet3D(nn.Module):
    """
    Simplified 3D UNet for patch-based training.
    
    - Fewer filters: 16 base (vs 32)
    - 3 levels (vs 4)
    - Less parameters to reduce overfitting
    """
    
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        base_filters: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(n_channels, base_filters, dropout)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._conv_block(base_filters, base_filters * 2, dropout)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4, dropout)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8, dropout)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4, dropout)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2, dropout)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters, dropout)
        
        # Output
        self.out = nn.Conv3d(base_filters, n_classes, 1)
    
    def _conv_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        """Convolutional block: Conv-BN-ReLU-Dropout-Conv-BN-ReLU"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Output
        return self.out(d1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = SimpleUNet3D(n_channels=1, n_classes=1, base_filters=16)
    
    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Compare with ComplexUNet3D
    from nodule_ai.model import ComplexUNet3D
    complex_model = ComplexUNet3D(n_channels=1, n_classes=1, base_filters=32)
    print(f"ComplexUNet3D parameters: {count_parameters(complex_model):,}")
