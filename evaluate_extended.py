#!/usr/bin/env python
"""
Evaluate Extended Phase 2 Model
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def evaluate_extended():
    """Evaluate Phase 2 extended model"""
    
    checkpoint = Path("checkpoints/complex_unet3d_focal_extended.pt")
    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ComplexUNet3D(n_channels=1, n_classes=1, base_filters=16)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()
    
    # Load dataset
    dataset = LIDCDataset(Path("data"), cache=False, target_shape=(160, 160, 160))
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    print("=" * 80)
    print("EVALUATE PHASE 2 EXTENDED")
    print("=" * 80)
    print()
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}
    
    with torch.no_grad():
        for batch in val_loader:
            volume = batch["volume"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.float32).numpy()
            
            logits = model(volume)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            sample_id = batch["sample_id"][0]
            results[sample_id] = {"mask": mask, "probs": probs}
    
    # Evaluate per threshold
    print(f"{'Thresh':<8} {'Dice':<10} {'Sens':<10} {'Spec':<10} {'Prec':<10} {'F1':<10}")
    print("-" * 60)
    
    for threshold in thresholds:
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for sample_id, data in results.items():
            mask = data["mask"].astype(bool)
            pred = (data["probs"] >= threshold).astype(bool)
            
            tp += np.sum(pred & mask)
            fp += np.sum(pred & ~mask)
            fn += np.sum(~pred & mask)
            tn += np.sum(~pred & ~mask)
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        sens = tp / (tp + fn + 1e-7)
        spec = tn / (tn + fp + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        f1 = 2 * prec * sens / (prec + sens + 1e-7)
        
        print(f"{threshold:<8.2f} {dice:<10.4f} {sens:<10.4f} {spec:<10.4f} {prec:<10.4f} {f1:<10.4f}")
    
    print()
    print("=" * 80)
    print("CHECKING PRODUCTION READINESS")
    print("=" * 80)
    
    # Use threshold 0.5 for final check
    threshold = 0.5
    tp, fp, fn, tn = 0, 0, 0, 0
    for sample_id, data in results.items():
        mask = data["mask"].astype(bool)
        pred = (data["probs"] >= threshold).astype(bool)
        tp += np.sum(pred & mask)
        fp += np.sum(pred & ~mask)
        fn += np.sum(~pred & mask)
        tn += np.sum(~pred & ~mask)
    
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    f1 = 2 * prec * sens / (prec + sens + 1e-7)
    
    print(f"Dice: {dice:.4f} (target ≥ 0.70) - {'✅' if dice >= 0.70 else '❌'}")
    print(f"Sensitivity: {sens:.4f} (target ≥ 0.85) - {'✅' if sens >= 0.85 else '❌'}")
    print(f"Precision: {prec:.4f} (target ≥ 0.60) - {'✅' if prec >= 0.60 else '❌'}")
    print(f"F1: {f1:.4f}")
    print()
    
    if dice >= 0.70 and sens >= 0.85 and prec >= 0.60:
        print("✅ MODEL READY FOR PRODUCTION")
    else:
        print("❌ MODEL NOT READY - NEEDS MORE IMPROVEMENT")


if __name__ == "__main__":
    evaluate_extended()
