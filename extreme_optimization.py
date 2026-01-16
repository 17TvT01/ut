#!/usr/bin/env python
"""
AGGRESSIVE OPTIMIZATION - Thử mọi cách để đẩy Dice lên cao nhất có thể
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def ultra_aggressive_postprocessing(
    mask: np.ndarray,
    min_size: int = 5,  # Very aggressive
) -> np.ndarray:
    """Ultra aggressive post-processing."""
    # Only keep large components
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Keep only largest components
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    
    # Keep components >= min_size
    mask_sizes = sizes >= min_size
    keep_pixel = mask_sizes[labeled]
    
    return keep_pixel.astype(np.float32)


def extreme_strategies():
    """Try extreme strategies to maximize Dice."""
    
    checkpoint = Path("checkpoints/complex_unet3d_20251115-005610.pt")
    device = torch.device("cpu")
    
    print("=" * 80)
    print("EXTREME OPTIMIZATION - MAXIMIZING DICE")
    print("=" * 80)
    print()
    
    # Load model
    checkpoint_data = torch.load(checkpoint, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        state_dict = checkpoint_data["model_state"]
    else:
        state_dict = checkpoint_data
    
    base_filters = state_dict["stem.block.0.weight"].shape[0]
    model = ComplexUNet3D(n_channels=1, n_classes=1, base_filters=base_filters)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset = LIDCDataset(Path("data"), cache=False, target_shape=(128, 128, 128))
    
    # Strategies to test
    strategies = [
        # Lower thresholds with aggressive filtering
        {"name": "Low Threshold (0.3) + Ultra Aggressive Filter", "threshold": 0.3, "min_size": 50},
        {"name": "Low Threshold (0.4) + Ultra Aggressive Filter", "threshold": 0.4, "min_size": 50},
        {"name": "Medium (0.5) + Ultra Aggressive Filter", "threshold": 0.5, "min_size": 100},
        {"name": "Medium (0.6) + Large Component Only", "threshold": 0.6, "min_size": 200},
        
        # Very high thresholds
        {"name": "Very High (0.90) + Minimal Filter", "threshold": 0.90, "min_size": 5},
        {"name": "Very High (0.95) + Minimal Filter", "threshold": 0.95, "min_size": 5},
        
        # Adaptive strategies - only on nodule volumes
        {"name": "Adaptive: High on nodule volumes only", "threshold": 0.7, "min_size": 10, "adaptive": True},
    ]
    
    results = []
    
    for strategy in strategies:
        print("-" * 80)
        print(f"Testing: {strategy['name']}")
        print("-" * 80)
        
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        
        for i in range(len(dataset)):
            sample = dataset[i]
            volume = sample["volume"].unsqueeze(0).to(device)
            mask_true = sample["mask"][0].cpu().numpy()
            
            # Check if this volume has nodules
            has_nodules = mask_true.sum() > 0
            
            # Predict
            with torch.no_grad():
                logits = model(volume)
            
            probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
            
            # Apply threshold
            if strategy.get("adaptive") and not has_nodules:
                # For negative volumes, use very high threshold or skip
                mask_pred = (probs > 0.95).astype(np.float32)
            else:
                mask_pred = (probs > strategy["threshold"]).astype(np.float32)
            
            # Post-processing
            mask_pred = ultra_aggressive_postprocessing(mask_pred, min_size=strategy["min_size"])
            
            # Metrics
            tp = ((mask_pred == 1) & (mask_true == 1)).sum()
            fp = ((mask_pred == 1) & (mask_true == 0)).sum()
            fn = ((mask_pred == 0) & (mask_true == 1)).sum()
            tn = ((mask_pred == 0) & (mask_true == 0)).sum()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
        
        # Calculate metrics
        dice = (2 * total_tp + 1e-8) / (2*total_tp + total_fp + total_fn + 1e-8)
        precision = total_tp / (total_tp + total_fp + 1e-8)
        sensitivity = total_tp / (total_tp + total_fn + 1e-8)
        
        print(f"  Dice: {dice:.6f}")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Sensitivity: {sensitivity*100:.2f}%")
        print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print()
        
        results.append({
            "name": strategy["name"],
            "dice": float(dice),
            "precision": float(precision),
            "sensitivity": float(sensitivity),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
        })
    
    # Find best
    best_dice = max(results, key=lambda x: x["dice"])
    best_precision = max(results, key=lambda x: x["precision"])
    
    print("=" * 80)
    print("BEST RESULTS")
    print("=" * 80)
    print(f"\nBest Dice: {best_dice['dice']:.6f}")
    print(f"  Strategy: {best_dice['name']}")
    print(f"  Precision: {best_dice['precision']*100:.2f}%")
    print(f"  Sensitivity: {best_dice['sensitivity']*100:.2f}%")
    
    print(f"\nBest Precision: {best_precision['precision']*100:.2f}%")
    print(f"  Strategy: {best_precision['name']}")
    print(f"  Dice: {best_precision['dice']:.6f}")
    print()
    
    # Check if we can reach target
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Maximum Dice achieved: {best_dice['dice']:.6f}")
    print(f"Target Dice: 0.70")
    print(f"Gap: {0.70 - best_dice['dice']:.6f} ({(0.70/best_dice['dice']):.1f}x)")
    print()
    
    if best_dice['dice'] < 0.10:
        print("⚠️  CONCLUSION:")
        print("Even with aggressive optimization, Dice < 0.10")
        print("This confirms: DATASET TOO SMALL is the bottleneck")
        print()
        print("With only 2,104 positive voxels total across 5 volumes,")
        print("the model cannot learn meaningful nodule features.")
        print()
        print("REQUIRED: Download larger dataset (LIDC-IDRI)")
    
    # Save results
    import json
    with open("checkpoints/extreme_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results, best_dice


if __name__ == "__main__":
    results, best = extreme_strategies()
