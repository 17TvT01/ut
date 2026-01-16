#!/usr/bin/env python
"""
Phase 1: Inference với Post-Processing
- Thay threshold từ 0.5 → 0.70
- Thêm morphological operations
- Lọc thành phần nhỏ
- Test kết quả
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def postprocess_predictions(
    binary_mask: np.ndarray,
    min_component_size: int = 10,
    use_morphology: bool = True,
) -> np.ndarray:
    """
    Post-processing predictions để giảm false positives.
    
    Args:
        binary_mask: Binary prediction (0 hoặc 1)
        min_component_size: Loại bỏ components < size này
        use_morphology: Áp dụng morphological operations
    """
    if ndimage is None:
        return binary_mask
    
    result = binary_mask.copy()
    
    # Step 1: Morphological operations
    if use_morphology:
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
        # Opening: Erosion -> Dilation (loại bỏ noise nhỏ)
        result = ndimage.binary_opening(result, structure=structure, iterations=1)
        # Closing: Dilation -> Erosion (làm liên thông)
        result = ndimage.binary_closing(result, structure=structure, iterations=1)
    
    # Step 2: Lọc thành phần theo kích thước
    if min_component_size > 1:
        labeled, num_features = ndimage.label(result)
        component_sizes = ndimage.sum(result, labeled, index=range(1, num_features + 1))
        
        # Giữ chỉ components lớn
        result_filtered = np.zeros_like(result)
        for comp_idx in range(1, num_features + 1):
            if component_sizes[comp_idx - 1] >= min_component_size:
                result_filtered[labeled == comp_idx] = 1
        result = result_filtered
    
    return result


def calculate_metrics_with_postprocessing(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    min_component_size: int = 10,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Tính metrics với post-processing.
    """
    # Convert predictions to binary
    probs = torch.sigmoid(predictions)
    pred_binary = (probs >= threshold).cpu().numpy().astype(np.uint8)
    target_np = targets.cpu().numpy().astype(np.uint8)
    
    # Post-process
    pred_processed = np.zeros_like(pred_binary)
    for i in range(pred_binary.shape[0]):
        pred_processed[i] = postprocess_predictions(
            pred_binary[i, 0],  # Get channel 0
            min_component_size=min_component_size,
            use_morphology=True
        )
    
    # Flatten và tính metrics
    pred_flat = pred_processed.flatten()
    target_flat = target_np.flatten()
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    sensitivity = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity + epsilon)
    
    return {
        "threshold": float(threshold),
        "min_component_size": int(min_component_size),
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
    
    print("=" * 80)
    print("PHASE 1: QUICK WINS - THRESHOLD + POST-PROCESSING")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Đang tải dataset...")
    dataset = LIDCDataset(data_dir, cache=False, target_shape=(160, 160, 160))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ Dataset: {len(dataset)} mẫu\n")
    
    # Load model
    print("Đang tải model...")
    model = load_checkpoint(checkpoint_path, device)
    print("✓ Model loaded\n")
    
    # Get all predictions
    print("Đang tính predictions...")
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
    print()
    
    # Test different configurations
    print("=" * 80)
    print("TESTING CONFIGURATIONS")
    print("=" * 80)
    print()
    
    configurations = [
        # (threshold, min_component_size, description)
        (0.50, 0, "Baseline (no post-processing)"),
        (0.50, 10, "Threshold 0.5 + filter size 10"),
        (0.60, 10, "Threshold 0.6 + filter size 10"),
        (0.65, 10, "Threshold 0.65 + filter size 10"),
        (0.70, 10, "Threshold 0.7 + filter size 10 ⭐"),
        (0.70, 20, "Threshold 0.7 + filter size 20"),
        (0.75, 10, "Threshold 0.75 + filter size 10"),
    ]
    
    results = []
    
    print(f"{'Config':<35} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1':<8}")
    print("-" * 95)
    
    for threshold, min_size, desc in configurations:
        metrics = calculate_metrics_with_postprocessing(
            predictions, targets,
            threshold=threshold,
            min_component_size=min_size
        )
        results.append({**metrics, "description": desc})
        
        print(
            f"{desc:<35} "
            f"{metrics['dice']:<8.4f} "
            f"{metrics['iou']:<8.4f} "
            f"{metrics['sensitivity']:<8.4f} "
            f"{metrics['specificity']:<8.4f} "
            f"{metrics['precision']:<8.4f} "
            f"{metrics['f1']:<8.4f}"
        )
    
    print()
    
    # Find best configuration
    print("=" * 80)
    print("🏆 BEST CONFIGURATIONS")
    print("=" * 80)
    print()
    
    best_dice_idx = max(range(len(results)), key=lambda i: results[i]['dice'])
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
    best_balanced_idx = max(
        range(len(results)),
        key=lambda i: results[i]['sensitivity'] * 0.5 + results[i]['specificity'] * 0.5
    )
    
    for label, idx in [("Best Dice", best_dice_idx), ("Best F1", best_f1_idx), ("Best Balanced", best_balanced_idx)]:
        r = results[idx]
        print(f"🎯 {label}:")
        print(f"   Config: {r['description']}")
        print(f"   Threshold: {r['threshold']}, Min Size: {r['min_component_size']}")
        print(f"   Dice: {r['dice']:.4f}, IoU: {r['iou']:.4f}, F1: {r['f1']:.4f}")
        print(f"   Sensitivity: {r['sensitivity']:.4f}, Specificity: {r['specificity']:.4f}")
        print(f"   Precision: {r['precision']:.4f}")
        print()
    
    # Save results
    output_file = Path("checkpoints/phase1_postprocessing_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Kết quả đã lưu: {output_file}")
    print()
    
    # Recommendation
    print("=" * 80)
    print("💡 KHUYẾN NGHỊ")
    print("=" * 80)
    best_r = results[best_f1_idx]
    print(f"""
✓ Sử dụng: Threshold = {best_r['threshold']}, Min Component Size = {best_r['min_component_size']}

Cải thiện:
  • Dice: 0.0007 → {best_r['dice']:.4f} ({best_r['dice']/0.0007:.1f}x)
  • Precision: 0.04% → {best_r['precision']*100:.2f}% ({best_r['precision']/0.0004:.0f}x)
  • F1: 0.0007 → {best_r['f1']:.4f}

Trạng thái: {'✅ ĐẠT YÊU CẦU' if best_r['dice'] >= 0.70 and best_r['precision'] >= 0.70 else '⚠️ CẢN RETRAIN'}
""")


if __name__ == "__main__":
    main()
