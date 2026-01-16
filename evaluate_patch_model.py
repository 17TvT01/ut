#!/usr/bin/env python
"""
Evaluate Patch-Based Model trên Full Volumes
Reconstruct predictions từ patches
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.simple_model import SimpleUNet3D


def sliding_window_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    stride: Tuple[int, int, int] = (32, 32, 32),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Perform sliding window inference on full volume.
    
    Args:
        model: Trained model
        volume: Input volume [1, D, H, W]
        patch_size: Patch dimensions
        stride: Stride for sliding window
        device: Device to run on
    
    Returns:
        Predicted mask [1, D, H, W]
    """
    model.eval()
    
    _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # Output volume to accumulate predictions
    output = torch.zeros_like(volume)
    counts = torch.zeros_like(volume)
    
    # Sliding window
    with torch.no_grad():
        for z in range(0, D - pd + 1, sd):
            for y in range(0, H - ph + 1, sh):
                for x in range(0, W - pw + 1, sw):
                    # Extract patch
                    patch = volume[:, z:z+pd, y:y+ph, x:x+pw]
                    patch = patch.to(device)
                    
                    # Predict
                    logits = model(patch)
                    pred = torch.sigmoid(logits)
                    
                    # Accumulate
                    output[:, z:z+pd, y:y+ph, x:x+pw] += pred.cpu()
                    counts[:, z:z+pd, y:y+ph, x:x+pw] += 1
    
    # Average overlapping predictions
    output = output / (counts + 1e-8)
    
    return output


def evaluate_patch_model(
    checkpoint_path: Path,
    data_dir: Path = Path("data"),
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    stride: Tuple[int, int, int] = (32, 32, 32),
    device: str = "cuda",
    threshold: float = 0.5,
):
    """
    Evaluate patch-based model on full volumes.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Data directory
        patch_size: Patch size used in training
        stride: Stride for sliding window inference
        device: Device to use
        threshold: Threshold for binary prediction
    """
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("EVALUATING PATCH-BASED MODEL")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Device: {device}")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Threshold: {threshold}")
    print()
    
    # Load model
    print("Loading model...")
    model = SimpleUNet3D(n_channels=1, n_classes=1, base_filters=16)
    state_dict = torch.load(checkpoint_path, map_location=device)
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
    
    # Evaluate each volume
    print("=" * 80)
    print("EVALUATING VOLUMES")
    print("=" * 80)
    print()
    
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_dice = 0.0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        volume = sample["volume"]  # [1, D, H, W]
        mask_true = sample["mask"]  # [1, D, H, W]
        
        print(f"Volume {i+1}/{len(dataset)}:")
        
        # Predict
        mask_pred = sliding_window_inference(
            model, volume, patch_size, stride, device
        )
        
        # Binarize
        mask_pred_binary = (mask_pred > threshold).float()
        
        # Compute metrics
        tp = ((mask_pred_binary == 1) & (mask_true == 1)).sum().item()
        fp = ((mask_pred_binary == 1) & (mask_true == 0)).sum().item()
        fn = ((mask_pred_binary == 0) & (mask_true == 1)).sum().item()
        tn = ((mask_pred_binary == 0) & (mask_true == 0)).sum().item()
        
        # Dice
        intersection = tp
        union = tp + fp + tp + fn  # 2*tp + fp + fn
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        total_dice += dice
        
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Dice: {dice:.6f}")
        print()
    
    # Overall metrics
    print("=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    
    # Dice
    avg_dice = total_dice / len(dataset)
    
    global_intersection = total_tp
    global_union = total_tp + total_fp + total_tp + total_fn
    global_dice = (2 * global_intersection + 1e-8) / (global_union + 1e-8)
    
    # Sensitivity, Specificity, Precision
    sensitivity = total_tp / (total_tp + total_fn + 1e-8)
    specificity = total_tn / (total_tn + total_fp + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    
    # IoU
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    
    # F1
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    
    print(f"Average Dice (per-volume): {avg_dice:.6f}")
    print(f"Global Dice: {global_dice:.6f}")
    print(f"IoU: {iou:.6f}")
    print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"F1 Score: {f1:.6f}")
    print()
    print(f"Confusion Matrix:")
    print(f"  TP: {total_tp:,}")
    print(f"  FP: {total_fp:,}")
    print(f"  FN: {total_fn:,}")
    print(f"  TN: {total_tn:,}")
    print()
    
    # Save results
    import json
    results = {
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "avg_dice": avg_dice,
        "global_dice": global_dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }
    
    results_path = checkpoint_path.with_suffix(".evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate patch-based model")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64], help="Patch size")
    parser.add_argument("--stride", type=int, nargs=3, default=[32, 32, 32], help="Stride")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    results = evaluate_patch_model(
        args.checkpoint,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        device=args.device,
        threshold=args.threshold,
    )
