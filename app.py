from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QImage, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nodule_ai.annotations import NoduleAnnotation, parse_annotation_xml
from nodule_ai.dicom import load_dicom_series
from nodule_ai.inference import analyze_nodules, infer_nodules, postprocess_nodules
from nodule_ai.model import ComplexUNet3D
from nodule_ai.trainer import TrainingConfig, train_model
from nodule_ai.storage import DatasetRegistry, ensure_local_path


class TrainingWorker(QObject):
    progress = pyqtSignal(int, int, float, float)
    message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            def callback(
                epoch: int,
                total: int,
                phase: str,
                train_loss: float | None,
                val_loss: float | None,
                message: str | None = None,
            ) -> None:
                if message:
                    self.message.emit(message)
                if phase == "metrics":
                    self.progress.emit(epoch, total, train_loss or 0.0, val_loss or 0.0)

            history = train_model(self.config, progress=callback)
            self.finished.emit(history)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.failed.emit(str(exc))


class DataSourcePrompt(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Them du lieu")
        self.resize(480, 140)
        self._selected_path: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Nhap duong dan hoac chon tu may:"))

        input_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        input_row.addWidget(self.path_edit)
        browse_dir_button = QPushButton("Chon thu muc...")
        browse_dir_button.clicked.connect(self._browse_directory)
        input_row.addWidget(browse_dir_button)
        browse_file_button = QPushButton("Chon file...")
        browse_file_button.clicked.connect(self._browse_file)
        input_row.addWidget(browse_file_button)
        layout.addLayout(input_row)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.ok_button = QPushButton("Dong y")
        self.ok_button.setEnabled(False)
        cancel_button = QPushButton("Huy")
        button_row.addWidget(self.ok_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        self.ok_button.clicked.connect(self._accept_if_valid)
        cancel_button.clicked.connect(self.reject)
        self.path_edit.textChanged.connect(self._on_text_changed)
        self.path_edit.returnPressed.connect(self._accept_if_valid)

    def selected_path(self) -> str:
        return self._selected_path or ""

    def _on_text_changed(self, value: str) -> None:
        self.ok_button.setEnabled(bool(value.strip()))

    def _accept_if_valid(self) -> None:
        value = self.path_edit.text().strip()
        if not value:
            return
        self._selected_path = value
        self.accept()

    def _browse_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chon thu muc du lieu")
        if path:
            self.path_edit.setText(path)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chon file du lieu",
            filter="ZIP Files (*.zip);;Tat ca (*.*)",
        )
        if path:
            self.path_edit.setText(path)


class DataSettingsDialog(QDialog):
    apply_analysis = pyqtSignal(str)
    apply_training = pyqtSignal(str)

    def __init__(self, registry: DatasetRegistry, cache_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.cache_root = cache_root
        self.setWindowTitle("Nguon du lieu")
        self.resize(520, 320)

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._update_button_states)
        self.list_widget.itemDoubleClicked.connect(self._on_use_for_analysis)
        layout.addWidget(self.list_widget)

        action_row = QHBoxLayout()
        self.add_button = QPushButton("Them...")
        self.remove_button = QPushButton("Xoa")
        action_row.addWidget(self.add_button)
        action_row.addWidget(self.remove_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        apply_row = QHBoxLayout()
        self.analysis_button = QPushButton("Dung cho phan tich")
        self.training_button = QPushButton("Dung cho huan luyen")
        apply_row.addWidget(self.analysis_button)
        apply_row.addWidget(self.training_button)
        layout.addLayout(apply_row)

        close_row = QHBoxLayout()
        close_row.addStretch()
        self.close_button = QPushButton("Dong")
        close_row.addWidget(self.close_button)
        layout.addLayout(close_row)

        self.add_button.clicked.connect(self._on_add)
        self.remove_button.clicked.connect(self._on_remove)
        self.analysis_button.clicked.connect(self._on_use_for_analysis)
        self.training_button.clicked.connect(self._on_use_for_training)
        self.close_button.clicked.connect(self.accept)

        self._refresh_list()

    def _refresh_list(self, select_path: Optional[str] = None) -> None:
        current = select_path or self._current_path()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        selected_item: Optional[QListWidgetItem] = None
        for entry in sorted(self.registry.entries):
            item = QListWidgetItem(entry)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.list_widget.addItem(item)
            if current and entry == current:
                selected_item = item
        if selected_item:
            self.list_widget.setCurrentItem(selected_item)
        self.list_widget.blockSignals(False)
        self._update_button_states()

    def _current_path(self) -> Optional[str]:
        item = self.list_widget.currentItem()
        if not item:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return str(data) if data else None

    def _on_add(self) -> None:
        prompt = DataSourcePrompt(self)
        if prompt.exec() != QDialog.DialogCode.Accepted:
            return
        source = prompt.selected_path().strip()
        if not source:
            return
        try:
            local_path = ensure_local_path(source, self.cache_root, force_dir=True)
        except Exception as exc:
            QMessageBox.warning(self, "Khong the tai du lieu", str(exc))
            return
        added = self.registry.add(local_path)
        self._refresh_list(str(local_path))
        if added:
            QMessageBox.information(self, "Hoan tat", f"Da san sang: {local_path}")

    def _on_remove(self) -> None:
        path_str = self._current_path()
        if not path_str:
            return
        if self.registry.remove(Path(path_str)):
            self._refresh_list()

    def _on_use_for_analysis(self) -> None:
        path_str = self._current_path()
        if not path_str:
            return
        self.apply_analysis.emit(path_str)

    def _on_use_for_training(self) -> None:
        path_str = self._current_path()
        if not path_str:
            return
        self.apply_training.emit(path_str)

    def _update_button_states(self) -> None:
        has_selection = self.list_widget.currentItem() is not None
        self.remove_button.setEnabled(has_selection)
        self.analysis_button.setEnabled(has_selection)
        self.training_button.setEnabled(has_selection)


class NoduleApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lung Nodule Assistant")
        self.resize(1280, 820)

        self.volume: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.summary: List[dict] = []
        self._checkpoint_path = Path("checkpoints/complex_unet3d.pt")
        self.inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_root = Path.home() / ".nodule_ai" / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.dataset_registry = DatasetRegistry(self.cache_root / "datasets.json")

        self._init_ui()
        self.training_thread: Optional[QThread] = None
        self.training_worker: Optional[TrainingWorker] = None

    # --------- UI setup ---------
    def _init_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.analysis_tab = QWidget()
        self.training_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Phân tích")
        self.tabs.addTab(self.training_tab, "Huấn luyện")

        self._build_analysis_tab()
        self._build_training_tab()
        self._setup_menu()

    def _setup_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Mo thu muc DICOM", self)
        open_action.triggered.connect(self._browse_dicom_dir)
        file_menu.addAction(open_action)

        settings_menu = menubar.addMenu("Cai dat")
        manage_action = QAction("Quan ly nguon du lieu...", self)
        manage_action.triggered.connect(self._open_data_settings)
        settings_menu.addAction(manage_action)

    # ----- analysis tab -----
    def _build_analysis_tab(self) -> None:
        layout = QVBoxLayout()

        source_group = QGroupBox("Nguồn dữ liệu")
        source_form = QFormLayout()

        self.dicom_path_edit = QLineEdit()
        dicom_button = QPushButton("Chon...")
        dicom_button.clicked.connect(self._browse_dicom_dir)
        dicom_box = QHBoxLayout()
        dicom_box.addWidget(self.dicom_path_edit)
        dicom_box.addWidget(dicom_button)
        dicom_container = QWidget()
        dicom_container.setLayout(dicom_box)
        source_form.addRow("Thu muc DICOM / ZIP", dicom_container)

        self.xml_path_edit = QLineEdit()
        xml_button = QPushButton("Chọn...")
        xml_button.clicked.connect(self._browse_xml_file)
        xml_box = QHBoxLayout()
        xml_box.addWidget(self.xml_path_edit)
        xml_box.addWidget(xml_button)
        xml_container = QWidget()
        xml_container.setLayout(xml_box)
        source_form.addRow("XML anot (tùy chọn)", xml_container)

        self.checkpoint_label = QLabel(str(self._checkpoint_path))
        self.checkpoint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        source_form.addRow("Checkpoint mô hình", self.checkpoint_label)

        source_group.setLayout(source_form)

        settings_group = QGroupBox("Thiết lập")
        settings_form = QFormLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_box = QHBoxLayout()
        threshold_box.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.50")
        threshold_box.addWidget(self.threshold_label)
        threshold_container = QWidget()
        threshold_container.setLayout(threshold_box)
        settings_form.addRow("Ngưỡng xác suất", threshold_container)

        self.min_voxels_spin = QSpinBox()
        self.min_voxels_spin.setRange(1, 5000)
        self.min_voxels_spin.setValue(50)
        settings_form.addRow("Voxel tối thiểu", self.min_voxels_spin)
        settings_group.setLayout(settings_form)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(source_group, stretch=2)
        controls_layout.addWidget(settings_group, stretch=1)
        layout.addLayout(controls_layout)


        self.run_button = QPushButton("Phân tích")
        self.run_button.clicked.connect(self._on_run_analysis)
        layout.addWidget(self.run_button)

        self.status_label = QLabel("Chưa chạy phân tích.")
        layout.addWidget(self.status_label)

        results_group = QGroupBox("Kết quả")
        results_layout = QHBoxLayout()
        results_group.setLayout(results_layout)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "Voxel", "Z", "Y", "X", "Malignancy"])
        self.table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.table, stretch=1)

        image_panel = QVBoxLayout()
        self.image_label = QLabel("Chưa có ảnh")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_panel.addWidget(self.image_label)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        slice_box = QHBoxLayout()
        self.slice_label = QLabel("Lát cắt: 0")
        slice_box.addWidget(self.slice_label)
        slice_box.addWidget(self.slice_slider)
        image_panel.addLayout(slice_box)

        results_layout.addLayout(image_panel, stretch=1)
        layout.addWidget(results_group)

        self.analysis_tab.setLayout(layout)

    # ----- training tab -----
    def _build_training_tab(self) -> None:
        layout = QVBoxLayout()

        data_group = QGroupBox("Dữ liệu huấn luyện")
        data_form = QFormLayout()
        self.train_data_edit = QLineEdit()
        data_button = QPushButton("Chon...")
        data_button.clicked.connect(self._browse_train_data)
        data_box = QHBoxLayout()
        data_box.addWidget(self.train_data_edit)
        data_box.addWidget(data_button)
        data_container = QWidget()
        data_container.setLayout(data_box)
        data_form.addRow("Thu muc du lieu", data_container)

        self.train_checkpoint_edit = QLineEdit(str(Path("checkpoints/complex_unet3d.pt")))
        self.train_checkpoint_edit.textChanged.connect(self._on_train_checkpoint_changed)
        out_button = QPushButton("Lưu tại...")
        out_button.clicked.connect(self._browse_save_checkpoint)
        out_box = QHBoxLayout()
        out_box.addWidget(self.train_checkpoint_edit)
        out_box.addWidget(out_button)
        out_container = QWidget()
        out_container.setLayout(out_box)
        data_form.addRow("Checkpoint xuất", out_container)

        data_group.setLayout(data_form)
        layout.addWidget(data_group)

        params_group = QGroupBox("Siêu tham số")
        params_form = QGridLayout()

        params_form.addWidget(QLabel("Epoch"), 0, 0)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 200)
        self.epoch_spin.setValue(20)
        params_form.addWidget(self.epoch_spin, 0, 1)

        params_form.addWidget(QLabel("Batch size"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 8)
        self.batch_spin.setValue(1)
        params_form.addWidget(self.batch_spin, 0, 3)

        params_form.addWidget(QLabel("Learning rate"), 1, 0)
        self.lr_edit = QLineEdit("0.001")
        params_form.addWidget(self.lr_edit, 1, 1)

        params_form.addWidget(QLabel("Val split"), 1, 2)
        self.val_edit = QLineEdit("0.2")
        params_form.addWidget(self.val_edit, 1, 3)

        params_form.addWidget(QLabel("Workers"), 2, 0)
        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(0, 8)
        self.worker_spin.setValue(0)
        params_form.addWidget(self.worker_spin, 2, 1)

        params_form.addWidget(QLabel("Device"), 2, 2)
        self.device_combo = QComboBox()
        devices = ["cuda", "cpu"]
        if not torch.cuda.is_available() and "cuda" in devices:
            devices.remove("cuda")
        self.device_combo.addItems(devices)
        params_form.addWidget(self.device_combo, 2, 3)

        params_group.setLayout(params_form)
        layout.addWidget(params_group)

        self.train_button = QPushButton("Bắt đầu huấn luyện")
        self.train_button.clicked.connect(self._on_start_training)
        layout.addWidget(self.train_button)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log)

        self.training_tab.setLayout(layout)

    def _remember_dataset(self, path: Path) -> None:
        try:
            resolved = Path(path).resolve()
        except Exception:
            resolved = Path(path)
        if resolved.is_dir():
            self.dataset_registry.add(resolved)

    def _open_data_settings(self) -> None:
        dialog = DataSettingsDialog(self.dataset_registry, self.cache_root, self)
        dialog.apply_analysis.connect(self._set_analysis_source)
        dialog.apply_training.connect(self._set_training_source)
        dialog.exec()

    def _set_analysis_source(self, path: str) -> None:
        if not path:
            return
        resolved = Path(path)
        self.dicom_path_edit.setText(str(resolved))
        self._remember_dataset(resolved)

    def _log_training(self, text: str) -> None:
        self.train_log.append(text)
        self.train_log.moveCursor(QTextCursor.MoveOperation.End)
        self.train_log.ensureCursorVisible()
        QApplication.processEvents()

    def _set_training_source(self, path: str) -> None:
        if not path:
            return
        resolved = Path(path)
        self.train_data_edit.setText(str(resolved))
        self._remember_dataset(resolved)

    def _set_checkpoint_path(self, path: Path) -> None:
        self._checkpoint_path = path
        if hasattr(self, "checkpoint_label"):
            self.checkpoint_label.setText(str(path))

    def _on_train_checkpoint_changed(self, value: str) -> None:
        value = value.strip()
        if value:
            self._set_checkpoint_path(Path(value))

    # --------- helpers ---------
    def _browse_dicom_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chon thu muc DICOM")
        if path:
            self.dicom_path_edit.setText(path)
            self._remember_dataset(Path(path))

    def _browse_xml_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn tệp XML", filter="XML Files (*.xml)")
        if path:
            self.xml_path_edit.setText(path)

    def _browse_train_data(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chon thu muc du lieu")
        if path:
            self.train_data_edit.setText(path)
            self._remember_dataset(Path(path))

    def _browse_save_checkpoint(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Chọn nơi lưu checkpoint",
            str(Path("checkpoints/complex_unet3d.pt")),
            "PyTorch checkpoint (*.pt *.pth)",
        )
        if path:
            self.train_checkpoint_edit.setText(path)

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"{value / 100:.2f}")

    def _on_slice_changed(self, value: int) -> None:
        self.slice_label.setText(f"Lát cắt: {value}")
        if self.volume is not None and self.mask is not None:
            pixmap = self._slice_to_pixmap(self.volume, self.mask, value)
            self.image_label.setPixmap(pixmap)

    def _on_run_analysis(self) -> None:
        dicom_source = self.dicom_path_edit.text().strip()
        if not dicom_source:
            QMessageBox.warning(self, "Thieu du lieu", "Hay nhap thu muc DICOM, file ZIP hoac link Google Drive.")
            return

        xml_text = self.xml_path_edit.text().strip()
        threshold = self.threshold_slider.value() / 100
        min_voxels = self.min_voxels_spin.value()

        try:
            self.status_label.setText("Dang xu ly...")
            QApplication.processEvents()
            dicom_path = ensure_local_path(dicom_source, self.cache_root, force_dir=True)
            self.dicom_path_edit.setText(str(dicom_path))
            self._remember_dataset(dicom_path)
            xml_path: Optional[Path] = None
            if xml_text:
                xml_path = ensure_local_path(xml_text, self.cache_root, force_dir=False)
                self.xml_path_edit.setText(str(xml_path))
            volume, mask, summary = self._run_pipeline(
                dicom_path,
                xml_path,
                threshold,
                min_voxels,
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            QMessageBox.critical(self, "Loi", str(exc))
            self.status_label.setText("Phan tich that bai.")
            return

        self.status_label.setText(f"Phat hien {len(summary)} not.")
        self.volume = volume
        self.mask = mask
        self.summary = summary

        self._populate_table(summary)
        if volume is not None and mask is not None:
            self.slice_slider.setEnabled(True)
            self.slice_slider.setMaximum(max(volume.shape[0] - 1, 0))
            self.slice_slider.setValue(volume.shape[0] // 2)
            pixmap = self._slice_to_pixmap(volume, mask, self.slice_slider.value())
            self.image_label.setPixmap(pixmap)
        else:
            self.slice_slider.setEnabled(False)
            self.image_label.setText("Khong co du lieu anh.")

    def _run_pipeline(
        self,
        dicom_source: Path,
        xml_path: Optional[Path],
        threshold: float,
        min_voxels: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[dict]]:
        volume_np, meta = load_dicom_series(dicom_source)

        annotations: List[NoduleAnnotation] = []
        if xml_path and xml_path.exists():
            annotations = parse_annotation_xml(xml_path)

        model = self._load_model()
        volume_tensor = (
            torch.from_numpy(volume_np)
            .unsqueeze(0)
            .to(
                self.inference_device,
                dtype=torch.float32,
                non_blocking=self.inference_device.type == "cuda",
            )
        )
        binary_mask = infer_nodules(model, volume_tensor, threshold=threshold)
        try:
            nodules = postprocess_nodules(binary_mask, min_voxels=min_voxels)
        except ImportError as exc:
            raise RuntimeError("Can cai scipy de hau xu ly.") from exc
        summary = analyze_nodules(nodules, annotations, meta)
        return volume_np, binary_mask, summary

    def _load_model(self) -> ComplexUNet3D:
        checkpoint_path = self._checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint '{checkpoint_path}'. Hãy huấn luyện mô hình trước."
            )

        params = {
            "n_channels": 1,
            "n_classes": 1,
            "base_filters": 32,
            "dropout": 0.1,
            "upsample_mode": "trilinear",
        }
        summary_path = checkpoint_path.with_suffix(".summary.json")
        if summary_path.exists():
            try:
                metadata = json.loads(summary_path.read_text(encoding="utf-8"))
                model_params = metadata.get("model_params") or {}
                for key in params:
                    if key in model_params:
                        params[key] = model_params[key]
            except Exception:
                pass

        device = self.inference_device
        model = ComplexUNet3D(**params).to(device)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            if "model_state" in state:
                state = state["model_state"]
            elif "state_dict" in state:
                state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint '{checkpoint_path}' không hợp lệ.")
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        return model

    def _populate_table(self, summary: List[dict]) -> None:
        self.table.setRowCount(len(summary))
        for row, item in enumerate(summary):
            centroid = item.get("centroid") or [0.0, 0.0, 0.0]
            values = [
                item.get("detected_id"),
                item.get("voxel_count"),
                round(float(centroid[0]), 2),
                round(float(centroid[1]), 2),
                round(float(centroid[2]), 2),
                item.get("malignancy_score"),
            ]
            for col, value in enumerate(values):
                text = "" if value is None else str(value)
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _slice_to_pixmap(self, volume: np.ndarray, mask: np.ndarray, index: int) -> QPixmap:
        base = (volume[index] * 255).clip(0, 255).astype(np.uint8)
        slice_mask = mask[index].astype(bool)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[slice_mask] = [255, 0, 0]
        rgb = np.ascontiguousarray(rgb)
        height, width, _ = rgb.shape
        bytes_per_line = 3 * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)

    def _on_start_training(self) -> None:
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.information(self, "Dang huan luyen", "Tien trinh khac dang chay.")
            return

        data_source = self.train_data_edit.text().strip()
        if not data_source:
            QMessageBox.warning(self, "Thieu du lieu", "Hay nhap thu muc LIDC hoac link Google Drive.")
            return

        try:
            data_dir = ensure_local_path(data_source, self.cache_root, force_dir=True)
            self.train_data_edit.setText(str(data_dir))
            self._remember_dataset(data_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Khong the truy cap du lieu", str(exc))
            return

        try:
            config = TrainingConfig(
                data_dir=data_dir,
                epochs=self.epoch_spin.value(),
                batch_size=self.batch_spin.value(),
                learning_rate=float(self.lr_edit.text()),
                val_split=float(self.val_edit.text()),
                num_workers=self.worker_spin.value(),
                seed=42,
                device=self.device_combo.currentText(),
                checkpoint=Path(self.train_checkpoint_edit.text()),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Tham so sai", str(exc))
            return

        self.train_button.setEnabled(False)
        self.train_log.clear()
        self._log_training("Bat dau huan luyen...")
        self._log_training(f"Su dung du lieu: {data_dir}")

        self.training_worker = TrainingWorker(config)
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self._on_training_progress)
        self.training_worker.message.connect(self._append_train_log)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self._cleanup_training_thread)
        self.training_thread.start()

    def _on_training_progress(self, epoch: int, total: int, train_loss: float, val_loss: float) -> None:
        self._log_training(f"Epoch {epoch}/{total}: train={train_loss:.4f}, val={val_loss:.4f}")

    def _append_train_log(self, message: str) -> None:
        self._log_training(message)

    def _on_training_finished(self, history: dict) -> None:
        checkpoint_path = Path(self.train_checkpoint_edit.text())
        self._set_checkpoint_path(checkpoint_path)
        self._log_training("\nHoan tat huan luyen!")
        self._log_training(f"Checkpoint luu tai: {checkpoint_path}")
        summary_path = checkpoint_path.with_suffix(".summary.json")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                device = summary.get("device", "unknown")
                amp_enabled = summary.get("use_amp")
                pin_memory = summary.get("pin_memory")
                msg = f"Thiet bi: {device}"
                if amp_enabled is not None:
                    msg += f" | AMP: {'bat' if amp_enabled else 'tat'}"
                if pin_memory is not None:
                    msg += f" | pin_memory: {'bat' if pin_memory else 'tat'}"
                self._log_training(msg)
            except Exception:
                pass
        self.train_button.setEnabled(True)

    def _on_training_failed(self, error: str) -> None:
        self._log_training(f"Loi: {error}")
        QMessageBox.critical(self, "Huan luyen that bai", error)
        self.train_button.setEnabled(True)

    def _cleanup_training_thread(self) -> None:
        if self.training_worker:
            self.training_worker.deleteLater()
            self.training_worker = None
        if self.training_thread:
            self.training_thread.deleteLater()
            self.training_thread = None


def main() -> None:
    app = QApplication(sys.argv)
    window = NoduleApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
