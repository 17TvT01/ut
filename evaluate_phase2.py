#!/usr/bin/env python
"""
Auto-evaluation sau training
- Load model vừa train
- Evaluate với threshold tối ưu
- So sánh với baseline
- Quyết định tiếp tục hay không
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model từ checkpoint."""
    model = ComplexUNet3D(
        n_channels=1,
        n_classes=1,
        base_filters=16,
        dropout=0.1,
        upsample_mode="trilinear",
    )
    
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded: {checkpoint_path}")
    else:
        print(f"❌ Not found: {checkpoint_path}")
        return None
    
    return model.to(device).eval()


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluate model trên validation set."""
    
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
    
    # Calculate metrics for different thresholds
    results = {}
    for threshold in [0.5, 0.6, 0.65, 0.7, 0.75]:
        probs = torch.sigmoid(predictions)
        pred_binary = (probs >= threshold).float()
        
        pred_flat = pred_binary.flatten()
        target_flat = targets.flatten()
        
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
        
        epsilon = 1e-6
        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        iou = (tp + epsilon) / (tp + fp + fn + epsilon)
        sensitivity = (tp + epsilon) / (tp + fn + epsilon)
        specificity = (tn + epsilon) / (tn + fp + epsilon)
        precision = (tp + epsilon) / (tp + fp + epsilon)
        f1 = (2 * precision * sensitivity) / (precision + sensitivity + epsilon)
        
        results[threshold] = {
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
    
    return results


def main():
    checkpoint = Path("checkpoints/complex_unet3d_focal.pt")
    data_dir = Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("AUTO-EVALUATION: PHASE 2 MODEL")
    print("=" * 80)
    print()
    
    # Check if model exists
    if not checkpoint.exists():
        print(f"⏳ Model training in progress...")
        print(f"   Expected path: {checkpoint}")
        print(f"   Please run again after training completes.")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint, device)
    if model is None:
        return
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = LIDCDataset(data_dir, cache=False, target_shape=(160, 160, 160))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ Dataset: {len(dataset)} samples\n")
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, dataloader, device)
    print()
    
    # Print results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    print(f"{'Thresh':<8} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1':<8}")
    print("-" * 80)
    
    best_f1_thresh = max(results.keys(), key=lambda t: results[t]['f1'])
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        marker = "⭐" if threshold == best_f1_thresh else "  "
        print(
            f"{threshold:<8.2f} "
            f"{r['dice']:<8.4f} "
            f"{r['iou']:<8.4f} "
            f"{r['sensitivity']:<8.4f} "
            f"{r['specificity']:<8.4f} "
            f"{r['precision']:<8.4f} "
            f"{r['f1']:<8.4f} {marker}"
        )
    
    print()
    
    # Compare with baseline
    baseline_file = Path("checkpoints/evaluation_metrics.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        best_result = results[best_f1_thresh]
        
        print("=" * 80)
        print("COMPARISON WITH BASELINE")
        print("=" * 80)
        print()
        print(f"{'Metric':<15} {'Baseline':<12} {'Phase2':<12} {'Improvement':<12}")
        print("-" * 80)
        
        metrics_to_compare = ['dice', 'precision', 'sensitivity', 'f1']
        
        for metric in metrics_to_compare:
            baseline_val = baseline.get(metric, 0)
            phase2_val = best_result.get(metric, 0)
            
            if baseline_val == 0:
                improvement = "N/A"
            else:
                improvement = f"{phase2_val / baseline_val:.1f}x"
            
            print(
                f"{metric.upper():<15} "
                f"{baseline_val:<12.4f} "
                f"{phase2_val:<12.4f} "
                f"{improvement:<12}"
            )
        
        print()
    
    # Check if meets requirements
    print("=" * 80)
    print("PRODUCTION READINESS")
    print("=" * 80)
    print()
    
    best_result = results[best_f1_thresh]
    checks = {
        "Dice ≥ 0.70": best_result['dice'] >= 0.70,
        "Precision ≥ 0.60": best_result['precision'] >= 0.60,
        "Sensitivity ≥ 0.85": best_result['sensitivity'] >= 0.85,
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    print()
    
    all_passed = all(checks.values())
    if all_passed:
        print("✅ MODEL READY FOR PRODUCTION")
    elif best_result['dice'] >= 0.60 and best_result['precision'] >= 0.40:
        print("⚠️ CLOSE - May proceed with caution or continue optimization")
    else:
        print("❌ MODEL NOT READY - Needs further optimization")
    
    print()
    
    # Save results
    output_file = Path("checkpoints/phase2_evaluation.json")
    with open(output_file, "w") as f:
        json.dump({
            "all_results": results,
            "best_threshold": best_f1_thresh,
            "best_result": best_result,
        }, f, indent=2)
    
    print(f"✓ Results saved: {output_file}")


if __name__ == "__main__":
    main()
