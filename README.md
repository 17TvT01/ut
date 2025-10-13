# Hướng dẫn sử dụng Lung Nodule Assistant

## 1. Giới thiệu
Ứng dụng "Lung Nodule Assistant" hỗ trợ phân tích loạt ảnh CT ngực định dạng DICOM để phát hiện nốt phổi, đồng thời cung cấp công cụ huấn luyện mô hình UNet3D trên bộ dữ liệu LIDC. Dự án bao gồm:

- Giao diện PyQt (`qt_app.py`) để chạy suy luận, xem lát cắt và bảng tổng hợp kết quả.
- Tập lệnh huấn luyện dòng lệnh (`train.py`) dùng PyTorch.
- Thư viện nội bộ `nodule_ai` chứa các thành phần xử lý DICOM, ánh xạ annotation và pipeline suy luận.

## 2. Yêu cầu hệ thống

- Python 3.10 trở lên (đã kiểm thử với Python 3.11).
- pip hoặc công cụ quản lý gói tương đương.
- GPU CUDA (tùy chọn) nếu muốn huấn luyện nhanh hơn; ứng dụng vẫn chạy trên CPU.
- Các thư viện liệt kê trong `requirements.txt`.

## 3. Cài đặt môi trường

```bash
python -m venv .venv
.venv\Scripts\activate   # Trên Windows
# source .venv/bin/activate  # Trên Linux/macOS
pip install -r requirements.txt
```

Lưu ý: bước hậu xử lý cần `scipy`. Thư viện này đã được khai báo trong `requirements.txt`.

## 4. Chuẩn bị dữ liệu đầu vào

Ứng dụng kỳ vọng cấu trúc dữ liệu kiểu LIDC:

```
data/
  └─ <study-id>/
       ├─ *.dcm        # Các lát cắt DICOM
       └─ *.xml        # Annotation tương ứng (tùy chọn)
```

Bạn có thể tải toàn bộ thư mục hoặc nén thành `.zip`. Trong giao diện, cả hai định dạng đều được hỗ trợ.

## 5. Chạy giao diện phân tích (PyQt)

1. Kích hoạt môi trường ảo (nếu có).
2. Khởi chạy ứng dụng:
   ```bash
   python qt_app.py
   ```
3. Trong tab **Phân tích**:
   - **Nguồn dữ liệu**: chọn thư mục DICOM hoặc file ZIP. Tùy chọn cung cấp file XML annotation và checkpoint mô hình (`.pt`/`.pth`).
   - **Thiết lập**:
     - *Ngưỡng xác suất*: trượt để điều chỉnh ngưỡng nhị phân hóa mask.
     - *Voxel tối thiểu*: loại bỏ các vùng có số voxel nhỏ hơn ngưỡng.
4. Nhấn **Phân tích**. Ứng dụng sẽ hiển thị:
   - Bảng tóm tắt nốt phát hiện được (ID, số voxel, tọa độ, mức ác tính ước lượng).
   - Khung xem ảnh với lát cắt ở giữa thể tích; có thể trượt để duyệt các lát.
5. Nếu cung cấp annotation XML, ứng dụng sẽ thử ghép cặp với nốt phát hiện để báo cáo khoảng cách.

## 6. Huấn luyện mô hình (CLI)

Tập lệnh `train.py` nhận tham số chi tiết cho quá trình huấn luyện:

```bash
python train.py <DATA_DIR> \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --val-split 0.2 \
    --checkpoint checkpoints/unet3d.pt \
    --num-workers 4 \
    --device cuda
```

Giải thích tham số chính (xem `train.py:7-44`):

- `<DATA_DIR>`: đường dẫn tới thư mục chứa các study dạng LIDC.
- `--epochs`, `--batch-size`, `--lr`, `--val-split`: siêu tham số huấn luyện.
- `--checkpoint`: vị trí lưu checkpoint tốt nhất; thư mục sẽ được tạo tự động.
- `--num-workers`: số worker DataLoader.
- `--device`: `cuda` hoặc `cpu`. Nếu không chỉ định, script sẽ tự chọn dựa trên tình trạng GPU.
- `--resume`: (tùy chọn) đường dẫn checkpoint để tiếp tục huấn luyện.

Khi hoàn tất, log sẽ in lịch sử loss và đường dẫn checkpoint tốt nhất. Một tệp `*.history.json` cũng được lưu kèm cùng checkpoint để tiện phân tích.

## 7. Thư viện `nodule_ai`

Các mô-đun quan trọng:

- `nodule_ai/dicom.py`: đọc loạt ảnh DICOM, chuẩn hóa HU, dựng mask annotation.
- `nodule_ai/inference.py`: suy luận, hậu xử lý mask bằng scipy, tạo báo cáo nốt.
- `nodule_ai/model.py`: định nghĩa UNet3D.
- `nodule_ai/dataset.py`: Dataset cho LIDC, sử dụng trong huấn luyện.
- `nodule_ai/annotations.py`: parser XML thành cấu trúc dataclass.

Có thể tái sử dụng các module này trong script riêng nếu cần.

## 8. Gợi ý thử nghiệm nhanh

- Dùng thư mục mẫu trong `data/` để chạy thử giao diện. Sau khi phân tích, điều chỉnh slider lát cắt để kiểm tra mask nốt.
- Thử nâng ngưỡng xác suất lên ~0.6 nếu phát hiện quá nhiều vùng nhiễu.
- Với huấn luyện, kiểm tra GPU trước khi đặt `--device cuda` bằng `python -c "import torch; print(torch.cuda.is_available())"`.

## 9. Hỗ trợ & phát triển thêm

- Để tích hợp thêm báo cáo tự động, xem `nodule_ai/report.py`.
- Khi cập nhật mô hình, đảm bảo checkpoint mới được trỏ tới trong giao diện.
- Đóng góp hoặc mở issue bằng cách mô tả rõ dataset, bước tái hiện lỗi và phiên bản thư viện.
