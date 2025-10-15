from __future__ import annotations

import json
import psutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
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
from nodule_ai.dataset import LIDCDataset
from nodule_ai.inference import analyze_nodules, infer_nodules, postprocess_nodules, extract_filtered_mask
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
        self._model_metadata: dict = {}
        self._default_target_shape: Tuple[int, int, int] | None = (160, 160, 160)
        self._labeled_mask: Optional[np.ndarray] = None
        self._selected_nodule_id: Optional[int] = None

        self._init_ui()
        self.training_thread: Optional[QThread] = None
        self.training_worker: Optional[TrainingWorker] = None
        self.monitor_timer: Optional[QTimer] = None

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

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["ID", "Voxel", "Z", "Y", "X", "Malignancy", "Color"])
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
        # Update tooltip to indicate multi-GPU support
        self.device_combo.setToolTip("Select device for training. Multi-GPU will be used automatically if multiple GPUs are available.")
        params_form.addWidget(self.device_combo, 2, 3)

        # Add GPU status display
        self.gpu_status_label = QLabel(self._get_gpu_status_text())
        self.gpu_status_label.setStyleSheet("font-weight: bold; color: green;" if torch.cuda.is_available() else "font-weight: bold; color: red;")
        params_form.addWidget(QLabel("GPU Status:"), 3, 0)
        params_form.addWidget(self.gpu_status_label, 3, 1, 1, 3)

        params_group.setLayout(params_form)
        layout.addWidget(params_group)

        self.train_button = QPushButton("Bắt đầu huấn luyện")
        self.train_button.clicked.connect(self._on_start_training)
        layout.addWidget(self.train_button)

        # Add GPU memory monitor during training
        monitor_group = QGroupBox("Theo dõi tài nguyên")
        monitor_layout = QHBoxLayout()
        monitor_group.setLayout(monitor_layout)

        self.gpu_memory_label = QLabel("GPU Memory: N/A")
        self.cpu_usage_label = QLabel("CPU: N/A")
        monitor_layout.addWidget(self.gpu_memory_label)
        monitor_layout.addWidget(self.cpu_usage_label)
        monitor_layout.addStretch()

        layout.addWidget(monitor_group)

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

    def _get_gpu_status_text(self) -> str:
        """Get GPU status information for display in UI."""
        if not torch.cuda.is_available():
            return "No GPU available - using CPU"

        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return f"1 GPU available: {gpu_name}"
            except Exception:
                return "1 GPU available"
        else:
            try:
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                gpu_name = gpu_names[0]
                if len(set(gpu_names)) == 1:
                    return f"{gpu_count} GPUs available: {gpu_name}"
                else:
                    return f"{gpu_count} GPUs available (mixed types)"
            except Exception:
                return f"{gpu_count} GPUs available"

    def _start_resource_monitoring(self) -> None:
        """Start monitoring GPU memory and CPU usage during training."""
        if self.monitor_timer:
            self.monitor_timer.stop()

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._update_resource_display)
        self.monitor_timer.start(2000)  # Update every 2 seconds

    def _stop_resource_monitoring(self) -> None:
        """Stop resource monitoring."""
        if self.monitor_timer:
            self.monitor_timer.stop()
            self.monitor_timer = None

    def _update_resource_display(self) -> None:
        """Update the resource usage display."""
        try:
            # GPU memory usage
            if torch.cuda.is_available():
                gpu_memory_used = 0
                gpu_memory_total = 0
                for i in range(torch.cuda.device_count()):
                    gpu_memory_used += torch.cuda.memory_allocated(i) / 1024**3  # GB
                    gpu_memory_total += torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                self.gpu_memory_label.setText(f"GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB")
            else:
                self.gpu_memory_label.setText("GPU Memory: N/A")

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / 1024**3
            memory_total_gb = memory.total / 1024**3
            self.cpu_usage_label.setText(f"CPU: {cpu_percent:.1f}% | RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f} GB")

        except Exception as e:
            # In case of any error, show N/A
            self.gpu_memory_label.setText("GPU Memory: N/A")
            self.cpu_usage_label.setText("CPU: N/A")

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
            pixmap = self._slice_to_pixmap(self.volume, self.mask, value, highlight_nodule_id=self._selected_nodule_id)
            self.image_label.setPixmap(pixmap)

    def _on_table_cell_clicked(self, row: int, column: int) -> None:
        """Navigate to the slice containing the clicked nodule."""
        if column == 0 and row < len(self.summary):  # ID column clicked
            nodule = self.summary[row]
            centroid = nodule.get("centroid", [0.0, 0.0, 0.0])
            z_slice = int(round(centroid[0]))  # Z-coordinate is slice index
            nodule_id = nodule.get("detected_id")

            # Ensure slice is within valid range
            if self.volume is not None:
                max_slice = self.volume.shape[0] - 1
                z_slice = max(0, min(z_slice, max_slice))

                # Set selected nodule for highlighting
                self._selected_nodule_id = nodule_id

                # Update slider and display
                self.slice_slider.blockSignals(True)  # Prevent recursive calls
                self.slice_slider.setValue(z_slice)
                self.slice_slider.blockSignals(False)

                # Update display with highlight
                pixmap = self._slice_to_pixmap(self.volume, self.mask, z_slice, highlight_nodule_id=nodule_id)
                self.image_label.setPixmap(pixmap)

                # Highlight the selected row
                self.table.selectRow(row)

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
        self._selected_nodule_id = None  # Reset selection when new analysis

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
        original_volume = volume_np
        self._labeled_mask = None
        model = self._load_model()
        target_shape = self._resolve_inference_shape(original_volume.shape)
        processed_volume = original_volume
        if target_shape and tuple(target_shape) != tuple(original_volume.shape):
            volume_tensor_cpu = torch.from_numpy(original_volume).unsqueeze(0)
            processed_volume = (
                LIDCDataset.downsample_volume(volume_tensor_cpu, target_shape)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        annotations: List[NoduleAnnotation] = []
        if xml_path and xml_path.exists():
            annotations = parse_annotation_xml(xml_path)

        volume_tensor = (
            torch.from_numpy(processed_volume.astype(np.float32))
            .unsqueeze(0)
            .to(
                self.inference_device,
                dtype=torch.float32,
                non_blocking=self.inference_device.type == "cuda",
            )
        )
        binary_mask = infer_nodules(
            model,
            volume_tensor,
            threshold=threshold,
            use_amp=self.inference_device.type == "cuda",
        )
        del volume_tensor
        if self.inference_device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        binary_mask = binary_mask.astype(np.uint8)
        if tuple(processed_volume.shape) != tuple(original_volume.shape):
            mask_tensor = (
                torch.from_numpy(binary_mask.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            mask_tensor = F.interpolate(
                mask_tensor,
                size=original_volume.shape,
                mode="nearest",
            )
            binary_mask = (
                mask_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            )
        try:
            nodules = postprocess_nodules(binary_mask, min_voxels=min_voxels)
        except ImportError as exc:
            raise RuntimeError("Can cai scipy de hau xu ly.") from exc
        display_mask, labeled_mask = extract_filtered_mask(binary_mask, nodules)
        self._labeled_mask = labeled_mask
        summary = analyze_nodules(nodules, annotations, meta)
        return original_volume, display_mask, summary

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
        summary_candidates = [
            checkpoint_path.with_suffix(".summary.json"),
            checkpoint_path.with_suffix(".history.summary.json"),
        ]
        metadata = None
        for candidate in summary_candidates:
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
                break
            except Exception:
                continue
        self._model_metadata = metadata or {}
        if metadata:
            model_params = metadata.get("model_params") or {}
            for key in params:
                if key in model_params:
                    params[key] = model_params[key]

        device = self.inference_device
        model = ComplexUNet3D(**params).to(device)

        # Enable multi-GPU inference if multiple GPUs are available and device is CUDA
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
            model = torch.nn.DataParallel(model)

        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            if "model_state" in state:
                state = state["model_state"]
            elif "state_dict" in state:
                state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid checkpoint '{checkpoint_path}'.")
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            hint = ""
            if metadata:
                original_filters = metadata.get("model_params", {}).get("base_filters")
                if original_filters is not None and original_filters != params.get("base_filters"):
                    hint = f" (checkpoint base_filters={original_filters}, current={params.get('base_filters')})"
            raise RuntimeError(f"Failed to load checkpoint: {exc}{hint}") from exc
        model = model.to(device)
        model.eval()
        return model

    def _resolve_inference_shape(self, volume_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        candidate = self._normalize_shape(self._model_metadata.get("target_shape"))
        if candidate:
            return candidate
        preprocessing = self._model_metadata.get("preprocessing")
        if isinstance(preprocessing, dict):
            prep_candidate = self._normalize_shape(preprocessing.get("target_shape"))
            if prep_candidate:
                return prep_candidate
        default_shape = self._normalize_shape(self._default_target_shape)
        if not default_shape:
            return None
        if self.inference_device.type != "cuda":
            return default_shape
        try:
            total_memory = torch.cuda.get_device_properties(self.inference_device).total_memory
        except Exception:
            total_memory = None
        if total_memory is None or total_memory <= 5 * (1024**3):
            return tuple(min(dim, tgt) for dim, tgt in zip(volume_shape, default_shape))
        return None

    @staticmethod
    def _normalize_shape(shape: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int]]:
        if shape is None:
            return None
        if isinstance(shape, tuple):
            candidate = list(shape)
        else:
            try:
                candidate = list(shape)  # type: ignore[arg-type]
            except TypeError:
                return None
        if len(candidate) != 3:
            return None
        try:
            normalized = tuple(max(1, int(value)) for value in candidate)
        except (TypeError, ValueError):
            return None
        return normalized

    def _populate_table(self, summary: List[dict]) -> None:
        self.table.setRowCount(len(summary))
        for row, item in enumerate(summary):
            centroid = item.get("centroid") or [0.0, 0.0, 0.0]
            malignancy = item.get("malignancy_score")

            # Determine color based on malignancy
            if malignancy is not None:
                if malignancy >= 4:
                    color_text = "🔴 High"
                elif malignancy >= 3:
                    color_text = "🟠 Medium"
                else:
                    color_text = "🟢 Low"
            else:
                color_text = "⚪ Unknown"

            values = [
                item.get("detected_id"),
                item.get("voxel_count"),
                round(float(centroid[0]), 2),
                round(float(centroid[1]), 2),
                round(float(centroid[2]), 2),
                item.get("malignancy_score"),
                color_text,
            ]
            for col, value in enumerate(values):
                text = "" if value is None else str(value)
                table_item = QTableWidgetItem(text)

                # Set background color based on malignancy
                if col == 6 and malignancy is not None:  # Color column
                    if malignancy >= 4:
                        table_item.setBackground(Qt.GlobalColor.red)
                    elif malignancy >= 3:
                        table_item.setBackground(Qt.GlobalColor.darkYellow)
                    else:
                        table_item.setBackground(Qt.GlobalColor.green)

                self.table.setItem(row, col, table_item)

        self.table.resizeColumnsToContents()
        # Connect click signal to navigate to nodule slice
        self.table.cellClicked.connect(self._on_table_cell_clicked)

    def _slice_to_pixmap(self, volume: np.ndarray, mask: np.ndarray, index: int, highlight_nodule_id: Optional[int] = None) -> QPixmap:
        from PIL import Image, ImageDraw, ImageFont

        base = (volume[index] * 255).clip(0, 255).astype(np.uint8)
        rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)
        pil_image = Image.fromarray(rgb).convert("RGBA")
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

        labeled_volume = None
        if isinstance(self._labeled_mask, np.ndarray) and self._labeled_mask.shape == mask.shape:
            labeled_volume = self._labeled_mask

        slice_nodules: List[dict] = []
        for nodule in self.summary:
            nodule_id = int(nodule.get("detected_id", 0) or 0)
            if nodule_id <= 0:
                continue
            if labeled_volume is not None and np.any(labeled_volume[index] == nodule_id):
                slice_nodules.append(nodule)
                continue
            centroid = nodule.get("centroid", [0, 0, 0])
            if int(round(centroid[0])) == index:
                slice_nodules.append(nodule)

        for nodule in slice_nodules:
            nodule_id = int(nodule.get("detected_id", 0) or 0)
            voxel_count = nodule.get("voxel_count", 0)

            if labeled_volume is not None:
                nodule_mask = labeled_volume[index] == nodule_id
            else:
                nodule_mask = mask[index] > 0.5

            if not np.any(nodule_mask):
                continue

            border_color = (255, 255, 0)
            boundary_mask = self._mask_boundary(nodule_mask)
            if np.any(boundary_mask):
                boundary_alpha = Image.fromarray((boundary_mask.astype(np.uint8) * 255), mode="L")
                border_layer = Image.new("RGBA", pil_image.size, (*border_color, 0))
                border_layer.putalpha(boundary_alpha)
                pil_image = Image.alpha_composite(pil_image, border_layer)
                draw = ImageDraw.Draw(pil_image)

            y_coords, x_coords = np.where(nodule_mask)
            if len(x_coords) > 0 and len(y_coords) > 0:
                min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
                min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
                draw.rectangle([min_x, min_y, max_x, max_y], outline=border_color, width=2)

                label_text = f"ID:{nodule_id}"
                malignancy = nodule.get("malignancy_score")
                if malignancy is not None:
                    label_text += f" M:{malignancy}"
                if voxel_count > 0:
                    label_text += f" V:{voxel_count}"

                text_x = min_x
                text_y = max(min_y - 20, 0)
                text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
                draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
                text_color = (255, 255, 255)
                if highlight_nodule_id is not None and nodule_id == highlight_nodule_id:
                    text_color = (255, 255, 0)
                draw.text((text_x, text_y), label_text, fill=text_color, font=font)

        pil_rgb = pil_image.convert("RGB")
        rgb_array = np.array(pil_rgb, dtype=np.uint8)
        rgb_array = np.ascontiguousarray(rgb_array)
        height, width, _ = rgb_array.shape
        bytes_per_line = 3 * width
        image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)

    @staticmethod
    def _mask_boundary(mask_slice: np.ndarray) -> np.ndarray:
        if mask_slice.ndim != 2:
            return np.zeros_like(mask_slice, dtype=bool)
        inner = (
            mask_slice
            & np.roll(mask_slice, 1, axis=0)
            & np.roll(mask_slice, -1, axis=0)
            & np.roll(mask_slice, 1, axis=1)
            & np.roll(mask_slice, -1, axis=1)
        )
        inner[0, :] = False
        inner[-1, :] = False
        inner[:, 0] = False
        inner[:, -1] = False
        return mask_slice & ~inner

    @staticmethod
    def _get_boundary_points(mask_slice: np.ndarray) -> List[Tuple[int, int]]:
        """Extract boundary points as polygon vertices for drawing."""
        boundary = NoduleApp._mask_boundary(mask_slice)
        if not np.any(boundary):
            return []

        # Find boundary pixels
        y_coords, x_coords = np.where(boundary)

        # Simple approach: find convex hull points for smoother boundary
        points = list(zip(x_coords, y_coords))

        if len(points) <= 2:
            return points

        # Sort points to create a rough polygon
        # Simple clockwise sorting around center
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        def angle_from_center(point):
            x, y = point
            return np.arctan2(y - center_y, x - center_x)

        points.sort(key=angle_from_center)
        return points

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

        # Start resource monitoring
        self._start_resource_monitoring()

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
                multi_gpu = summary.get("multi_gpu")
                gpu_count = summary.get("gpu_count")
                msg = f"Thiet bi: {device}"
                if amp_enabled is not None:
                    msg += f" | AMP: {'bat' if amp_enabled else 'tat'}"
                if pin_memory is not None:
                    msg += f" | pin_memory: {'bat' if pin_memory else 'tat'}"
                if multi_gpu is not None and multi_gpu:
                    msg += f" | Multi-GPU: {'bat' if multi_gpu else 'tat'} ({gpu_count} GPUs)"
                self._log_training(msg)
            except Exception:
                pass
        self._stop_resource_monitoring()
        self.gpu_memory_label.setText("GPU Memory: N/A")
        self.cpu_usage_label.setText("CPU: N/A")
        self.train_button.setEnabled(True)

    def _on_training_failed(self, error: str) -> None:
        self._stop_resource_monitoring()
        self.gpu_memory_label.setText("GPU Memory: N/A")
        self.cpu_usage_label.setText("CPU: N/A")
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
