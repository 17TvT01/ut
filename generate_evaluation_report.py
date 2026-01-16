#!/usr/bin/env python
"""
Tạo báo cáo đánh giá mô hình chi tiết dưới dạng markdown
"""

import json
from pathlib import Path

# Đọc kết quả đánh giá
metrics_file = Path("checkpoints/evaluation_metrics.json")
with open(metrics_file, "r") as f:
    metrics = json.load(f)

# Tạo báo cáo markdown
report = f"""# 📋 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHÁT HIỆN NỐTS PHỔI

## 📊 Tóm tắt chỉ số chính

| Chỉ số | Giá trị | Phần trăm |
|--------|--------|----------|
| **Dice Coefficient** | {metrics['dice_coefficient']:.4f} | {metrics['dice_coefficient']*100:.2f}% |
| **IoU (Intersection over Union)** | {metrics['iou']:.4f} | {metrics['iou']*100:.2f}% |
| **F1 Score** | {metrics['f1_score']:.4f} | {metrics['f1_score']*100:.2f}% |

---

## 🎯 Khả năng phát hiện nốts (Sensitivity / Recall)

**Sensitivity: {metrics['sensitivity_recall']:.4f} ({metrics['sensitivity_recall']*100:.2f}%)**

✅ **Ý nghĩa:** Mô hình phát hiện được **{metrics['sensitivity_recall']*100:.2f}%** nốts thực tế

- **Dương tính thực (TP):** {metrics['true_positives']} nốts được phát hiện đúng
- **Âm tính giả (FN):** {metrics['false_negatives']} nốts bị bỏ sót

> **💡 Kết luận:** Sensitivity rất cao, hầu như không bỏ sót nốts. Tuy nhiên, đây là vấn đề vì model đang dự đoán quá nhiều false positives.

---

## 🔍 Độ chuyên biệt (Specificity)

**Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)**

✅ **Ý nghĩa:** Mô hình xác định đúng **{metrics['specificity']*100:.2f}%** các voxel không phải nốt

- **Âm tính thực (TN):** {metrics['true_negatives']} voxel được xác định đúng là không phải nốt
- **Dương tính giả (FP):** {metrics['false_positives']} voxel được sai dự đoán là nốt

> **⚠️ Kết luận:** Specificity thấp, model đang dự đoán quá nhiều voxel là nốt (false positives). Điều này khiến Precision rất thấp.

---

## 📌 Độ chính xác của dự đoán (Precision)

**Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)**

✅ **Ý nghĩa:** Trong các voxel được dự đoán là nốt, chỉ **{metrics['precision']*100:.2f}%** thực sự là nốt

> **❌ Kết luận:** Precision rất thấp, model tạo ra rất nhiều false alarms.

---

## 📈 Chi tiết Confusion Matrix

| Loại | Số lượng |
|------|---------|
| True Positives (TP) | {metrics['true_positives']} |
| False Positives (FP) | {metrics['false_positives']} |
| False Negatives (FN) | {metrics['false_negatives']} |
| True Negatives (TN) | {metrics['true_negatives']} |
| **Tổng Positive Thực** | {metrics['true_positives'] + metrics['false_negatives']} |
| **Tổng Positive Dự đoán** | {metrics['true_positives'] + metrics['false_positives']} |

---

## 🔴 Vấn đề chính được xác định

### 1. **Quá nhiều False Positives**
- Model dự đoán {metrics['false_positives']:,} voxel sai là nốt
- Tỷ lệ False Positive Rate: {metrics['false_positives']/(metrics['false_positives'] + metrics['true_negatives'])*100:.2f}%

### 2. **Mất cân bằng giữa Sensitivity và Precision**
- Sensitivity cao ({metrics['sensitivity_recall']*100:.2f}%) nhưng Precision thấp ({metrics['precision']*100:.2f}%)
- Điều này gợi ý model threshold có thể cần điều chỉnh hoặc có vấn đề trong huấn luyện

### 3. **Dice và IoU rất thấp**
- Dice: {metrics['dice_coefficient']:.4f} (mục tiêu tốt: > 0.7)
- IoU: {metrics['iou']:.4f} (mục tiêu tốt: > 0.6)

---

## 💡 Khuyến nghị cải thiện

1. **Điều chỉnh Threshold:** 
   - Hiện tại sử dụng threshold = 0.5
   - Thử threshold cao hơn (0.7-0.8) để giảm False Positives

2. **Kiểm tra Dice Loss:**
   - Xem xét sử dụng weighted loss hoặc Focal Loss
   - Điều chỉnh class weight nếu dataset không cân bằng

3. **Phân tích Dataset:**
   - Kiểm tra chất lượng labels trong training data
   - Xem xét augmentation strategy

4. **Model Architecture:**
   - Xem xét các mô hình khác (V-Net, 3D ResNet)
   - Tăng model capacity nếu có đủ dữ liệu

5. **Huấn luyện lại:**
   - Sử dụng Learning Rate Scheduler
   - Tăng số lượng epochs
   - Cân bằng dữ liệu training

---

## ⚙️ Thông tin Huấn luyện

- **Model:** ComplexUNet3D
- **Device:** CUDA
- **Input Shape:** (160, 160, 160)
- **Số mẫu đánh giá:** 5
- **Threshold:** 0.5

---

*Báo cáo được tạo tự động từ evaluate_model.py*
"""

# Lưu báo cáo
report_file = Path("checkpoints/EVALUATION_REPORT.md")
report_file.write_text(report, encoding="utf-8")
print(f"✓ Báo cáo đã lưu: {report_file}")
print("\n" + report)
