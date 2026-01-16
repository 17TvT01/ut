#!/usr/bin/env python
"""
Script để tìm threshold tối ưu nhằm cân bằng giữa Sensitivity và Precision
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Tính toán các chỉ số với threshold cụ thể."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs >= threshold).float()
    
    pred_flat = pred_binary.flatten()
    target_flat = targets.flatten()
    
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    sensitivity = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity + epsilon)
    
    return {
        "threshold": float(threshold),
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model từ checkpoint."""
    model = ComplexUNet3D(
        n_channels=1,
        n_classes=1,
        base_filters=16,
        dropout=0.1,
        upsample_mode="trilinear",
    )
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
    
    return model.to(device).eval()


def main():
    data_dir = Path("data")
    checkpoint_path = Path("checkpoints/complex_unet3d.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Đang tải dataset và model...")
    dataset = LIDCDataset(data_dir, cache=False, target_shape=(160, 160, 160))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_checkpoint(checkpoint_path, device)
    
    print(f"✓ Dataset: {len(dataset)} mẫu\n")
    
    # Lấy tất cả predictions
    print("Đang tính toán predictions...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            volume = batch["volume"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(volume)
            
            all_predictions.append(logits.detach().cpu())
            all_targets.append(mask.detach().cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Kiểm tra các thresholds khác nhau
    print("\nKiểm tra các thresholds...")
    print("=" * 100)
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    print(f"{'Thresh':<8} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1':<8} {'TP':<8} {'FP':<10} {'FN':<8}")
    print("-" * 100)
    
    for threshold in thresholds:
        metrics = calculate_metrics(predictions, targets, threshold=threshold)
        results.append(metrics)
        
        print(
            f"{metrics['threshold']:<8.2f} "
            f"{metrics['dice']:<8.4f} "
            f"{metrics['iou']:<8.4f} "
            f"{metrics['sensitivity']:<8.4f} "
            f"{metrics['specificity']:<8.4f} "
            f"{metrics['precision']:<8.4f} "
            f"{metrics['f1']:<8.4f} "
            f"{metrics['tp']:<8} "
            f"{metrics['fp']:<10} "
            f"{metrics['fn']:<8}"
        )
    
    # Tìm threshold tối ưu dựa trên F1 Score
    best_f1_idx = np.argmax([r['f1'] for r in results])
    best_f1 = results[best_f1_idx]
    
    # Tìm threshold với Dice tối ưu
    best_dice_idx = np.argmax([r['dice'] for r in results])
    best_dice = results[best_dice_idx]
    
    # Tìm threshold cân bằng Sensitivity-Precision (Youden's J statistic)
    youden_scores = [
        r['sensitivity'] + r['specificity'] - 1 
        for r in results
    ]
    best_youden_idx = np.argmax(youden_scores)
    best_youden = results[best_youden_idx]
    
    print("\n" + "=" * 100)
    print("\n🏆 KẾT QUẢ TỐI ƯU:")
    print()
    
    print("📈 Threshold tốt nhất dựa trên F1 Score:")
    print(f"  • Threshold: {best_f1['threshold']:.2f}")
    print(f"  • F1 Score: {best_f1['f1']:.4f}")
    print(f"  • Dice: {best_f1['dice']:.4f}, IoU: {best_f1['iou']:.4f}")
    print(f"  • Sensitivity: {best_f1['sensitivity']:.4f}, Precision: {best_f1['precision']:.4f}")
    print()
    
    print("🎯 Threshold tốt nhất dựa trên Dice:")
    print(f"  • Threshold: {best_dice['threshold']:.2f}")
    print(f"  • Dice: {best_dice['dice']:.4f}")
    print(f"  • IoU: {best_dice['iou']:.4f}, F1: {best_dice['f1']:.4f}")
    print(f"  • Sensitivity: {best_dice['sensitivity']:.4f}, Precision: {best_dice['precision']:.4f}")
    print()
    
    print("⚖️ Threshold cân bằng (Youden):")
    print(f"  • Threshold: {best_youden['threshold']:.2f}")
    print(f"  • Youden Score: {youden_scores[best_youden_idx]:.4f}")
    print(f"  • Sensitivity: {best_youden['sensitivity']:.4f}, Specificity: {best_youden['specificity']:.4f}")
    print(f"  • Dice: {best_youden['dice']:.4f}, Precision: {best_youden['precision']:.4f}")
    print()
    
    # Lưu kết quả
    output_file = Path("checkpoints/threshold_analysis.json")
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "best_f1": best_f1,
            "best_dice": best_dice,
            "best_youden": best_youden,
            "youden_scores": [float(s) for s in youden_scores],
        }, f, indent=2)
    
    print(f"✓ Kết quả đã lưu: {output_file}")


if __name__ == "__main__":
    main()
