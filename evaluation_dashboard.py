#!/usr/bin/env python
"""
Evaluation Dashboard - So sánh models
- Baseline (original Dice Loss)
- Phase 1 (Threshold 0.75 + post-processing)
- Phase 2 (Focal Loss)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results(checkpoint_prefix: str) -> Dict:
    """Load evaluation results từ files."""
    results = {}
    
    # Load baseline
    baseline_file = Path("checkpoints/evaluation_metrics.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            results['baseline'] = json.load(f)
    
    # Load Phase 1
    phase1_file = Path("checkpoints/phase1_postprocessing_results.json")
    if phase1_file.exists():
        with open(phase1_file) as f:
            all_configs = json.load(f)
            # Find best one
            results['phase1'] = max(all_configs, key=lambda x: x['f1'])
    
    return results


def create_comparison_table(results: Dict) -> str:
    """Tạo bảng so sánh các models."""
    
    headers = ["Model", "Dice", "IoU", "Sensitivity", "Specificity", "Precision", "F1 Score"]
    
    # Collect rows
    rows = []
    for model_name, metrics in results.items():
        row = [
            model_name.upper(),
            f"{metrics.get('dice', 0):.4f}",
            f"{metrics.get('iou', 0):.4f}",
            f"{metrics.get('sensitivity', 0):.4f}",
            f"{metrics.get('specificity', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
        ]
        rows.append(row)
    
    # Create table
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    
    table = "\n"
    table += "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐\n"
    table += "│ " + " │ ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " │\n"
    table += "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤\n"
    
    for row in rows:
        table += "│ " + " │ ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " │\n"
    
    table += "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘\n"
    
    return table


def create_improvement_summary(results: Dict) -> str:
    """Tạo summary về improvements."""
    
    baseline = results.get('baseline', {})
    phase1 = results.get('phase1', {})
    
    if not baseline or not phase1:
        return "\n⚠️ Không đủ dữ liệu để so sánh\n"
    
    summary = "\n"
    summary += "=" * 80 + "\n"
    summary += "📊 IMPROVEMENT SUMMARY\n"
    summary += "=" * 80 + "\n\n"
    
    metrics_to_track = ['dice', 'precision', 'f1', 'sensitivity', 'specificity']
    
    summary += "IMPROVEMENTS:\n\n"
    for metric in metrics_to_track:
        baseline_val = baseline.get(metric, 0)
        phase1_val = phase1.get(metric, 0)
        
        if baseline_val == 0:
            improvement = "N/A"
        else:
            improvement = f"{phase1_val / baseline_val:.1f}x"
        
        summary += f"  {metric.upper():<15} {baseline_val:<10.4f} → {phase1_val:<10.4f}  ({improvement})\n"
    
    summary += "\n"
    
    # Check if meets requirements
    summary += "STATUS:\n"
    if phase1['dice'] >= 0.70 and phase1['precision'] >= 0.70:
        summary += "  ✅ MEETS PRODUCTION REQUIREMENTS\n"
    elif phase1['dice'] >= 0.60 and phase1['precision'] >= 0.50:
        summary += "  ⚠️ CLOSE - Minor improvements needed\n"
    else:
        summary += "  ❌ NEEDS MORE WORK - Continue optimization\n"
    
    summary += "\n"
    
    return summary


def main():
    results = load_results("")
    
    if not results:
        print("❌ No results found. Please run evaluation scripts first.")
        return
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON DASHBOARD")
    print("=" * 80)
    
    # Print comparison table
    table = create_comparison_table(results)
    print(table)
    
    # Print improvements
    summary = create_improvement_summary(results)
    print(summary)
    
    # Recommendations
    print("=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80 + "\n")
    
    phase1 = results.get('phase1', {})
    
    if phase1.get('dice', 0) < 0.70:
        print("""
Phase 1 Quick Fix không đạt mục tiêu. Bước tiếp theo:

1. PHASE 2 (RETRAIN):
   ✓ Chạy: python phase2_focal_loss_training.py
   ✓ Focal Loss sẽ giảm False Positives
   ✓ Thêm Learning Rate Scheduler & Early Stopping
   ✓ Tăng epochs từ 20 → 100

2. EXPECTED IMPROVEMENTS:
   • Dice: 0.0033 → 0.10-0.30 (30-90x)
   • Precision: 0.17% → 5-15% (30-90x)
   • Sensitivity: 35% → 75-85%

3. TIMELINE:
   • Focal Loss training: 30-60 phút
   • Evaluation: 5 phút
   • Total: ~1-2 giờ
""")
    else:
        print("✅ Phase 1 đã đạt mục tiêu!")
        print("Tiếp tục với Phase 2 để cải thiện thêm.")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
