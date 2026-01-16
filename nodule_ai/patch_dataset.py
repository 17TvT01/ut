#!/usr/bin/env python
"""
Patch-Based Dataset - Extract patches from volumes
Chiến lược: Tạo nhiều training samples từ ít volumes
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import random

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from nodule_ai.dataset import LIDCDataset


class PatchDataset(Dataset):
    """
    Extract patches from full volumes.
    
    Strategy:
    - Positive patches: crop around nodule centers
    - Negative patches: random crops from background
    - Heavy augmentation on patches
    """
    
    def __init__(
        self,
        base_dataset: LIDCDataset,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        num_positive_per_volume: int = 10,
        num_negative_per_volume: int = 30,
        augment: bool = True,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.num_positive = num_positive_per_volume
        self.num_negative = num_negative_per_volume
        self.augment = augment
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Extract all patches from base dataset
        self.patches: List[Dict[str, torch.Tensor]] = []
        print(f"Extracting patches from {len(base_dataset)} volumes...")
        
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            volume = sample["volume"]  # [1, D, H, W]
            mask = sample["mask"]      # [1, D, H, W]
            
            # Positive patches (around nodules)
            positive_patches = self._extract_positive_patches(volume, mask)
            self.patches.extend(positive_patches)
            
            # Negative patches (random from background)
            negative_patches = self._extract_negative_patches(volume, mask)
            self.patches.extend(negative_patches)
        
        print(f"✓ Extracted {len(self.patches)} patches")
        print(f"  - Positive: {sum(1 for p in self.patches if p['label'] == 1)}")
        print(f"  - Negative: {sum(1 for p in self.patches if p['label'] == 0)}")
    
    def _extract_positive_patches(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract patches centered on nodule voxels."""
        patches = []
        
        # Find nodule voxels
        nodule_coords = torch.nonzero(mask[0] > 0.5, as_tuple=False)  # [N, 3]
        
        if len(nodule_coords) == 0:
            # No nodules - skip
            return patches
        
        # Sample random nodule voxels as patch centers
        num_patches = min(self.num_positive, len(nodule_coords))
        indices = torch.randperm(len(nodule_coords))[:num_patches]
        selected_coords = nodule_coords[indices]
        
        for coord in selected_coords:
            z, y, x = coord.tolist()
            patch_vol, patch_mask = self._crop_patch(volume, mask, z, y, x)
            
            if patch_vol is not None:
                patches.append({
                    "volume": patch_vol,
                    "mask": patch_mask,
                    "label": 1,  # Positive patch
                })
        
        return patches
    
    def _extract_negative_patches(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract random patches from background (non-nodule regions)."""
        patches = []
        
        _, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        # Safety margins
        min_z, max_z = pd // 2, D - pd // 2
        min_y, max_y = ph // 2, H - ph // 2
        min_x, max_x = pw // 2, W - pw // 2
        
        attempts = 0
        max_attempts = self.num_negative * 5
        
        while len(patches) < self.num_negative and attempts < max_attempts:
            # Random center
            z = random.randint(min_z, max_z)
            y = random.randint(min_y, max_y)
            x = random.randint(min_x, max_x)
            
            # Check if this patch contains nodule
            z_start = max(0, z - pd // 2)
            z_end = min(D, z + pd // 2)
            y_start = max(0, y - ph // 2)
            y_end = min(H, y + ph // 2)
            x_start = max(0, x - pw // 2)
            x_end = min(W, x + pw // 2)
            
            patch_mask_region = mask[0, z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Only accept if patch has very few or no nodule voxels
            if patch_mask_region.sum() < 10:  # Allow some noise
                patch_vol, patch_mask = self._crop_patch(volume, mask, z, y, x)
                
                if patch_vol is not None:
                    patches.append({
                        "volume": patch_vol,
                        "mask": patch_mask,
                        "label": 0,  # Negative patch
                    })
            
            attempts += 1
        
        return patches
    
    def _crop_patch(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor,
        center_z: int,
        center_y: int,
        center_x: int,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Crop a patch centered at (center_z, center_y, center_x)."""
        _, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        # Calculate boundaries
        z_start = center_z - pd // 2
        z_end = z_start + pd
        y_start = center_y - ph // 2
        y_end = y_start + ph
        x_start = center_x - pw // 2
        x_end = x_start + pw
        
        # Check bounds
        if z_start < 0 or z_end > D or y_start < 0 or y_end > H or x_start < 0 or x_end > W:
            # Out of bounds - pad
            patch_vol = volume[:, 
                              max(0, z_start):min(D, z_end),
                              max(0, y_start):min(H, y_end),
                              max(0, x_start):min(W, x_end)]
            patch_mask = mask[:,
                             max(0, z_start):min(D, z_end),
                             max(0, y_start):min(H, y_end),
                             max(0, x_start):min(W, x_end)]
            
            # Pad to patch_size
            target_shape = (1, pd, ph, pw)
            if patch_vol.shape != target_shape:
                patch_vol = F.pad(
                    patch_vol,
                    (
                        max(0, -x_start), max(0, x_end - W),
                        max(0, -y_start), max(0, y_end - H),
                        max(0, -z_start), max(0, z_end - D),
                    ),
                    mode='constant',
                    value=0,
                )
                patch_mask = F.pad(
                    patch_mask,
                    (
                        max(0, -x_start), max(0, x_end - W),
                        max(0, -y_start), max(0, y_end - H),
                        max(0, -z_start), max(0, z_end - D),
                    ),
                    mode='constant',
                    value=0,
                )
        else:
            patch_vol = volume[:, z_start:z_end, y_start:y_end, x_start:x_end]
            patch_mask = mask[:, z_start:z_end, y_start:y_end, x_start:x_end]
        
        return patch_vol, patch_mask
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch = self.patches[idx]
        
        volume = patch["volume"].clone()
        mask = patch["mask"].clone()
        
        # Augmentation
        if self.augment:
            volume, mask = self._augment(volume, mask)
        
        return {
            "volume": volume,
            "mask": mask,
            "label": patch["label"],
        }
    
    def _augment(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations."""
        
        # 1. Random flip (50% each axis)
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1])  # Flip D
            mask = torch.flip(mask, dims=[1])
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[2])  # Flip H
            mask = torch.flip(mask, dims=[2])
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[3])  # Flip W
            mask = torch.flip(mask, dims=[3])
        
        # 2. Random 90-degree rotation in axial plane (30% chance)
        if random.random() > 0.7:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            volume = torch.rot90(volume, k, dims=[2, 3])
            mask = torch.rot90(mask, k, dims=[2, 3])
        
        # 3. Intensity scaling (±20%)
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            volume = volume * scale
        
        # 4. Add Gaussian noise (20% chance)
        if random.random() > 0.8:
            noise = torch.randn_like(volume) * 0.05
            volume = volume + noise
        
        # 5. Brightness shift (30% chance)
        if random.random() > 0.7:
            shift = random.uniform(-0.1, 0.1)
            volume = volume + shift
        
        # Clamp volume to valid range
        volume = torch.clamp(volume, -1, 1)
        
        return volume, mask


def create_patch_datasets(
    data_dir: Path | str,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    num_positive_per_volume: int = 10,
    num_negative_per_volume: int = 30,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[PatchDataset, PatchDataset]:
    """
    Create train and validation patch datasets.
    
    Returns:
        (train_dataset, val_dataset)
    """
    # Load base dataset
    base_dataset = LIDCDataset(data_dir, cache=False, target_shape=(128, 128, 128))
    
    # Split into train/val
    val_size = max(1, int(len(base_dataset) * val_split))
    train_size = len(base_dataset) - val_size
    
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(seed)
    train_base, val_base = random_split(base_dataset, [train_size, val_size], generator=generator)
    
    # Create patch datasets
    print("\n" + "=" * 80)
    print("CREATING TRAIN PATCH DATASET")
    print("=" * 80)
    train_patches = PatchDataset(
        train_base,
        patch_size=patch_size,
        num_positive_per_volume=num_positive_per_volume,
        num_negative_per_volume=num_negative_per_volume,
        augment=True,
        seed=seed,
    )
    
    print("\n" + "=" * 80)
    print("CREATING VAL PATCH DATASET")
    print("=" * 80)
    val_patches = PatchDataset(
        val_base,
        patch_size=patch_size,
        num_positive_per_volume=5,  # Fewer patches for validation
        num_negative_per_volume=15,
        augment=False,  # No augmentation for validation
        seed=seed + 1,
    )
    
    return train_patches, val_patches


if __name__ == "__main__":
    # Test
    train_dataset, val_dataset = create_patch_datasets(
        Path("data"),
        patch_size=(64, 64, 64),
        num_positive_per_volume=15,
        num_negative_per_volume=45,
    )
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Train patches: {len(train_dataset)}")
    print(f"Val patches: {len(val_dataset)}")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"\nSample volume shape: {sample['volume'].shape}")
    print(f"Sample mask shape: {sample['mask'].shape}")
    print(f"Sample label: {sample['label']}")
