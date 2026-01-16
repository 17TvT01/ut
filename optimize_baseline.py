#!/usr/bin/env python
"""
QUICK WIN STRATEGY - Optimize existing baseline model
Thay vì train model mới (chậm), optimize model cũ với:
1. Better threshold selection
2. Advanced post-processing
3. Ensemble different thresholds
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def advanced_postprocessing(
    mask: np.ndarray,
    min_size: int = 10,
    morphology: bool = True,
) -> np.ndarray:
    """
    Advanced post-processing.
    
    Args:
        mask: Binary mask [D, H, W]
        min_size: Minimum component size
        morphology: Apply morphological operations
    
    Returns:
        Processed mask
    """
    # 1. Remove small components
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    mask_sizes = sizes < min_size
    remove_pixel = mask_sizes[labeled]
    labeled[remove_pixel] = 0
    
    mask = (labeled > 0).astype(np.float32)
    
    # 2. Morphological operations
    if morphology:
        # Binary closing - fill holes
        struct = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_closing(mask, structure=struct, iterations=2)
        
        # Binary opening - remove noise
        mask = ndimage.binary_opening(mask, structure=struct, iterations=1)
    
    return mask.astype(np.float32)


def multi_threshold_ensemble(
    logits: torch.Tensor,
    thresholds: list = [0.3, 0.5, 0.7],
    weights: list = [0.2, 0.6, 0.2],
) -> torch.Tensor:
    """
    Ensemble predictions from multiple thresholds.
    
    Args:
        logits: Model logits
        thresholds: List of thresholds
        weights: Weights for each threshold
    
    Returns:
        Weighted ensemble prediction
    """
    probs = torch.sigmoid(logits)
    
    ensemble = torch.zeros_like(probs)
    for threshold, weight in zip(thresholds, weights):
        mask = (probs > threshold).float()
        ensemble += mask * weight
    
    return ensemble


def optimize_baseline_model(
    checkpoint_path: Path,
    data_dir: Path = Path("data"),
    device: str = "cuda",
):
    """
    Optimize baseline model với better post-processing.
    """
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("OPTIMIZING BASELINE MODEL")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Device: {device}")
    print()
    
    # Load model
    print("Loading model...")
    
    # Try loading to detect base_filters
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        state_dict = checkpoint_data["model_state"]
    else:
        state_dict = checkpoint_data
    
    # Detect base_filters from checkpoint
    base_filters = 32  # Default
    if "stem.block.0.weight" in state_dict:
        stem_filters = state_dict["stem.block.0.weight"].shape[0]
        base_filters = stem_filters
    
    print(f"Detected base_filters: {base_filters}")
    
    model = ComplexUNet3D(n_channels=1, n_classes=1, base_filters=base_filters)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = LIDCDataset(data_dir, cache=False, target_shape=(128, 128, 128))
    print(f"✓ Dataset loaded: {len(dataset)} volumes")
    print()
    
    # Test different strategies
    strategies = [
        {"name": "Baseline (threshold=0.5)", "threshold": 0.5, "postproc": False, "ensemble": False},
        {"name": "High Threshold (0.75)", "threshold": 0.75, "postproc": False, "ensemble": False},
        {"name": "High Threshold + Postproc", "threshold": 0.75, "postproc": True, "ensemble": False},
        {"name": "Ensemble (0.3,0.5,0.7)", "threshold": None, "postproc": True, "ensemble": True},
        {"name": "Very High (0.85) + Postproc", "threshold": 0.85, "postproc": True, "ensemble": False},
    ]
    
    results = []
    
    for strategy in strategies:
        print("=" * 80)
        print(f"STRATEGY: {strategy['name']}")
        print("=" * 80)
        print()
        
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_dice = 0.0
        
        for i in range(len(dataset)):
            sample = dataset[i]
            volume = sample["volume"].unsqueeze(0).to(device)  # [1, 1, D, H, W]
            mask_true = sample["mask"][0].cpu().numpy()  # [D, H, W]
            
            # Predict
            with torch.no_grad():
                logits = model(volume)
            
            logits = logits[0, 0].cpu()  # [D, H, W]
            
            # Apply strategy
            if strategy["ensemble"]:
                mask_pred = multi_threshold_ensemble(
                    logits.unsqueeze(0),
                    thresholds=[0.3, 0.5, 0.7],
                    weights=[0.2, 0.6, 0.2],
                )[0].numpy()
                mask_pred = (mask_pred > 0.5).astype(np.float32)
            else:
                probs = torch.sigmoid(logits).numpy()
                mask_pred = (probs > strategy["threshold"]).astype(np.float32)
            
            # Post-processing
            if strategy["postproc"]:
                mask_pred = advanced_postprocessing(mask_pred, min_size=10, morphology=True)
            
            # Compute metrics
            tp = ((mask_pred == 1) & (mask_true == 1)).sum()
            fp = ((mask_pred == 1) & (mask_true == 0)).sum()
            fn = ((mask_pred == 0) & (mask_true == 1)).sum()
            tn = ((mask_pred == 0) & (mask_true == 0)).sum()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
            # Dice
            intersection = tp
            dice = (2 * intersection + 1e-8) / (2*tp + fp + fn + 1e-8)
            total_dice += dice
        
        # Overall metrics
        avg_dice = total_dice / len(dataset)
        global_dice = (2 * total_tp + 1e-8) / (2*total_tp + total_fp + total_fn + 1e-8)
        
        sensitivity = total_tp / (total_tp + total_fn + 1e-8)
        specificity = total_tn / (total_tn + total_fp + 1e-8)
        precision = total_tp / (total_tp + total_fp + 1e-8)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
        
        result = {
            "strategy": strategy["name"],
            "avg_dice": float(avg_dice),
            "global_dice": float(global_dice),
            "iou": float(iou),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "tn": int(total_tn),
        }
        
        results.append(result)
        
        print(f"Global Dice: {global_dice:.6f}")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print()
    
    # Compare results
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    
    print(f"{'Strategy':<35} {'Dice':<10} {'Precision':<12} {'Sensitivity':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['strategy']:<35} {r['global_dice']:<10.6f} "
              f"{r['precision']*100:<12.2f} {r['sensitivity']*100:<12.2f}")
    
    # Find best
    best = max(results, key=lambda x: x['global_dice'])
    
    print()
    print("=" * 80)
    print("BEST STRATEGY")
    print("=" * 80)
    print(f"Strategy: {best['strategy']}")
    print(f"Global Dice: {best['global_dice']:.6f}")
    print(f"Precision: {best['precision']*100:.2f}%")
    print(f"Sensitivity: {best['sensitivity']*100:.2f}%")
    print()
    
    # Save results
    import json
    results_path = checkpoint_path.parent / "optimized_baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved: {results_path}")
    
    return results, best


if __name__ == "__main__":
    # Use baseline model
    checkpoint = Path("checkpoints/complex_unet3d_20251115-005610.pt")
    
    if not checkpoint.exists():
        # Try other checkpoints
        checkpoint = Path("checkpoints/complex_unet3d.pt")
    
    if checkpoint.exists():
        results, best = optimize_baseline_model(checkpoint)
    else:
        print(f"✗ Checkpoint not found: {checkpoint}")
        print("Available checkpoints:")
        for ckpt in Path("checkpoints").glob("*.pt"):
            print(f"  - {ckpt.name}")
