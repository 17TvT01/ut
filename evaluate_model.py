#!/usr/bin/env python
"""
Evaluation script để tính toán các chỉ số Chi tiết:
- Dice Coefficient
- IoU (Intersection over Union)
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá từ predictions và targets.
    
    Args:
        predictions: Output từ model (logits hoặc probabilities)
        targets: Ground truth masks
        threshold: Threshold để chuyển probabilities thành binary predictions
        epsilon: Giá trị nhỏ để tránh chia cho 0
    
    Returns:
        Dictionary chứa các chỉ số
    """
    # Convert predictions thành binary
    probs = torch.sigmoid(predictions)
    pred_binary = (probs >= threshold).float()
    
    # Flatten tensors để tính toán
    pred_flat = pred_binary.flatten()
    target_flat = targets.flatten()
    
    # Tính True Positives, False Positives, False Negatives, True Negatives
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # Dice Coefficient
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    
    # IoU (Intersection over Union)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    
    # Sensitivity (Recall) - tỷ lệ nốt phổi được phát hiện
    sensitivity = (tp + epsilon) / (tp + fn + epsilon)
    
    # Specificity - tỷ lệ không phải nốt phổi được xác định đúng
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    
    # Precision - trong số những gì được dự đoán là nốt, bao nhiêu là đúng
    precision = (tp + epsilon) / (tp + fp + epsilon)
    
    # F1 Score
    f1 = (2 * precision * sensitivity) / (precision + sensitivity + epsilon)
    
    return {
        "dice_coefficient": float(dice),
        "iou": float(iou),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Đánh giá model trên toàn bộ dataset.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            volume = batch["volume"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(volume)
            
            all_predictions.append(logits.detach().cpu())
            all_targets.append(mask.detach().cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, threshold=threshold)
    
    return metrics


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model từ checkpoint."""
    model = ComplexUNet3D(
        n_channels=1,
        n_classes=1,
        base_filters=16,
        dropout=0.1,
        upsample_mode="trilinear",
    )
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Xử lý các định dạng checkpoint khác nhau
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                # Thử load trực tiếp nếu là full state dict
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Đã tải checkpoint từ: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint không tìm thấy: {checkpoint_path}")
    
    return model.to(device).eval()


def main():
    # Configuration
    data_dir = Path("data")
    checkpoint_path = Path("checkpoints/complex_unet3d.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print("-" * 60)
    
    # Load dataset
    print("Đang tải dataset...")
    dataset = LIDCDataset(data_dir, cache=False, target_shape=(160, 160, 160))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ Dataset: {len(dataset)} mẫu")
    print()
    
    # Load model
    print("Đang tải model...")
    model = load_checkpoint(checkpoint_path, device)
    print()
    
    # Evaluate
    print("Đang đánh giá model...")
    metrics = evaluate_model(model, dataloader, device, use_amp=True, threshold=0.5)
    print()
    
    # Display results
    print("=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    print()
    
    print("📊 CHỈ SỐ CHÍNH:")
    print(f"  • Dice Coefficient: {metrics['dice_coefficient']:.4f}")
    print(f"  • IoU: {metrics['iou']:.4f}")
    print(f"  • F1 Score: {metrics['f1_score']:.4f}")
    print()
    
    print("🎯 PHÁT HIỆN NỐTS (Sensitivity):")
    print(f"  • Sensitivity (Recall): {metrics['sensitivity_recall']:.4f} ({metrics['sensitivity_recall']*100:.2f}%)")
    print(f"    → Phát hiện được {metrics['sensitivity_recall']*100:.2f}% nốts thực tế")
    print()
    
    print("🔍 CHUYÊN BIỆT (Specificity):")
    print(f"  • Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"    → Xác định đúng {metrics['specificity']*100:.2f}% voxels không phải nốt")
    print()
    
    print("📌 ĐỘ CHÍNH XÁC (Precision):")
    print(f"  • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"    → Trong số dự đoán là nốt, {metrics['precision']*100:.2f}% là chính xác")
    print()
    
    print("📈 CHI TIẾT MATRICE:")
    print(f"  • True Positives: {metrics['true_positives']}")
    print(f"  • False Positives: {metrics['false_positives']}")
    print(f"  • False Negatives: {metrics['false_negatives']}")
    print(f"  • True Negatives: {metrics['true_negatives']}")
    print()
    
    # Save results
    results_file = Path("checkpoints/evaluation_metrics.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Kết quả đã lưu: {results_file}")
    
    return metrics


if __name__ == "__main__":
    main()
