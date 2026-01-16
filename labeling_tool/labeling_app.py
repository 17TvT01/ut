"""
Interactive Labeling Application for CT Scan Nodule Annotation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QFileDialog,
    QListWidget,
    QMessageBox,
    QShortcut,
    QSpinBox,
    QComboBox,
)


class NoduleLabel:
    """Represents a single nodule annotation"""

    def __init__(self, bbox: Optional[Tuple[int, int, int, int]], slice_idx: int, label_type: str = "nodule"):
        self.bbox = bbox  # (x, y, w, h) - for rectangle/ellipse
        self.slice_idx = slice_idx
        self.label_type = label_type  # nodule, non-nodule, suspicious
        self.shape = "rectangle"  # rectangle, circle, ellipse, polygon, freehand
        self.circle = None  # (x, y, r) - for circle
        self.points = None  # [(x, y), ...] - for polygon/freehand

    def to_dict(self) -> dict:
        result = {
            "bbox": list(self.bbox) if self.bbox else None,
            "slice_idx": self.slice_idx,
            "label_type": self.label_type,
            "shape": self.shape,
        }
        if self.circle:
            result["circle"] = list(self.circle)
        if self.points:
            result["points"] = self.points
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "NoduleLabel":
        bbox = tuple(data["bbox"]) if data.get("bbox") else None
        label = cls(
            bbox=bbox,
            slice_idx=data["slice_idx"],
            label_type=data.get("label_type", "nodule"),
        )
        label.shape = data.get("shape", "rectangle")
        if "circle" in data:
            label.circle = tuple(data["circle"])
        if "points" in data:
            label.points = data["points"]
        return label


class ImageViewer(QLabel):
    """Custom widget for displaying and annotating CT images"""

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.drawing = False
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.current_rect = QRect()
        self.image: Optional[QPixmap] = None
        self.annotations: List[Dict] = []  # List of annotations with shape info
        self.polygon_points: List[QPoint] = []  # For polygon drawing
        self.scale_factor = 1.0
        self.draw_mode = "rectangle"  # rectangle, circle, ellipse, polygon, freehand

    def set_image(self, np_image: np.ndarray):
        """Set the image from numpy array"""
        # Normalize to 0-255
        img_normalized = ((np_image - np_image.min()) / (np_image.max() - np_image.min() + 1e-8) * 255).astype(
            np.uint8
        )
        # Convert to RGB
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image = QPixmap.fromImage(q_image)
        self.scale_factor = min(800 / w, 800 / h, 1.0)
        self.setPixmap(self.image.scaled(int(w * self.scale_factor), int(h * self.scale_factor)))

    def set_annotations(self, annotations: List[Dict]):
        """Set existing annotations for this slice"""
        self.annotations = annotations
        self.update_display()

    def set_draw_mode(self, mode: str):
        """Set the drawing mode"""
        self.draw_mode = mode
        self.polygon_points = []

    def update_display(self):
        """Redraw image with annotations"""
        if self.image is None:
            return

        pixmap = self.image.copy()
        painter = QPainter(pixmap)

        # Draw existing annotations
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for annotation in self.annotations:
            shape_type = annotation.get("shape", "rectangle")
            
            if shape_type == "rectangle":
                x, y, w, h = annotation["bbox"]
                painter.drawRect(int(x), int(y), int(w), int(h))
            elif shape_type == "circle":
                x, y, r = annotation["circle"]
                painter.drawEllipse(QPoint(int(x), int(y)), int(r), int(r))
            elif shape_type == "ellipse":
                x, y, w, h = annotation["bbox"]
                painter.drawEllipse(int(x), int(y), int(w), int(h))
            elif shape_type == "polygon":
                from PyQt5.QtGui import QPolygon
                points = [QPoint(int(p[0]), int(p[1])) for p in annotation["points"]]
                painter.drawPolygon(QPolygon(points))
            elif shape_type == "freehand":
                from PyQt5.QtGui import QPolygon
                points = [QPoint(int(p[0]), int(p[1])) for p in annotation["points"]]
                painter.drawPolyline(QPolygon(points))

        # Draw current drawing shape
        if self.drawing or self.polygon_points:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            
            if self.draw_mode == "rectangle":
                painter.drawRect(self.current_rect)
            elif self.draw_mode == "circle":
                center = self.start_point
                radius = int(np.sqrt((self.current_point.x() - center.x())**2 + 
                                   (self.current_point.y() - center.y())**2))
                painter.drawEllipse(center, radius, radius)
            elif self.draw_mode == "ellipse":
                painter.drawEllipse(self.current_rect)
            elif self.draw_mode == "polygon" and len(self.polygon_points) > 0:
                from PyQt5.QtGui import QPolygon
                # Draw completed segments
                if len(self.polygon_points) > 1:
                    painter.drawPolyline(QPolygon(self.polygon_points))
                # Draw line to current mouse position
                if self.drawing:
                    painter.drawLine(self.polygon_points[-1], self.current_point)
            elif self.draw_mode == "freehand" and len(self.polygon_points) > 0:
                from PyQt5.QtGui import QPolygon
                painter.drawPolyline(QPolygon(self.polygon_points))

        painter.end()
        self.setPixmap(pixmap.scaled(int(pixmap.width() * self.scale_factor), int(pixmap.height() * self.scale_factor)))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image:
            # Get position relative to the actual image, not the scaled display
            img_pos = self._get_image_position(event.pos())
            if img_pos is None:
                return
            
            if self.draw_mode == "polygon":
                # Add point to polygon
                self.polygon_points.append(img_pos)
                self.drawing = True
                self.current_point = img_pos
                self.update_display()
            else:
                self.start_point = img_pos
                self.drawing = True

    def mouseMoveEvent(self, event):
        if self.image:
            img_pos = self._get_image_position(event.pos())
            if img_pos is None:
                return
            
            self.current_point = img_pos
            
            if self.drawing:
                if self.draw_mode in ["rectangle", "ellipse"]:
                    self.current_rect = QRect(self.start_point, self.current_point).normalized()
                elif self.draw_mode == "freehand":
                    self.polygon_points.append(self.current_point)
                
                self.update_display()

    def _get_image_position(self, widget_pos: QPoint) -> Optional[QPoint]:
        """Convert widget position to image coordinates"""
        if self.image is None:
            return None
        
        # Get the actual displayed pixmap size
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        
        # Calculate the offset (centering)
        widget_width = self.width()
        widget_height = self.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        
        offset_x = (widget_width - pixmap_width) // 2
        offset_y = (widget_height - pixmap_height) // 2
        
        # Adjust for offset
        adjusted_x = widget_pos.x() - offset_x
        adjusted_y = widget_pos.y() - offset_y
        
        # Check if within bounds
        if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= pixmap_width or adjusted_y >= pixmap_height:
            return None
        
        # Scale to original image coordinates
        img_x = int(adjusted_x / self.scale_factor)
        img_y = int(adjusted_y / self.scale_factor)
        
        return QPoint(img_x, img_y)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            if self.draw_mode == "polygon":
                # Don't finish on release, wait for double-click or right-click
                return
                
            self.drawing = False
            
            if self.draw_mode == "rectangle":
                if self.current_rect.width() > 5 and self.current_rect.height() > 5:
                    bbox = (
                        self.current_rect.x(),
                        self.current_rect.y(),
                        self.current_rect.width(),
                        self.current_rect.height(),
                    )
                    self.annotations.append({"shape": "rectangle", "bbox": bbox})
                    self.update_display()
                    
            elif self.draw_mode == "circle":
                center = self.start_point
                radius = int(np.sqrt((self.current_point.x() - center.x())**2 + 
                                   (self.current_point.y() - center.y())**2))
                if radius > 5:
                    self.annotations.append({
                        "shape": "circle",
                        "circle": (center.x(), center.y(), radius)
                    })
                    self.update_display()
                    
            elif self.draw_mode == "ellipse":
                if self.current_rect.width() > 5 and self.current_rect.height() > 5:
                    bbox = (
                        self.current_rect.x(),
                        self.current_rect.y(),
                        self.current_rect.width(),
                        self.current_rect.height(),
                    )
                    self.annotations.append({"shape": "ellipse", "bbox": bbox})
                    self.update_display()
                    
            elif self.draw_mode == "freehand":
                if len(self.polygon_points) > 5:
                    points = [(p.x(), p.y()) for p in self.polygon_points]
                    self.annotations.append({"shape": "freehand", "points": points})
                self.polygon_points = []
                self.update_display()

    def mouseDoubleClickEvent(self, event):
        """Finish polygon drawing on double-click"""
        if self.draw_mode == "polygon" and len(self.polygon_points) > 2:
            points = [(p.x(), p.y()) for p in self.polygon_points]
            self.annotations.append({"shape": "polygon", "points": points})
            self.polygon_points = []
            self.drawing = False
            self.update_display()

    def contextMenuEvent(self, event):
        """Finish polygon on right-click"""
        if self.draw_mode == "polygon" and len(self.polygon_points) > 2:
            points = [(p.x(), p.y()) for p in self.polygon_points]
            self.annotations.append({"shape": "polygon", "points": points})
            self.polygon_points = []
            self.drawing = False
            self.update_display()


class LabelingApp(QMainWindow):
    """Main labeling application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nodule Labeling Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.dicom_files: List[Path] = []
        self.current_slice_idx = 0
        self.volume: Optional[np.ndarray] = None
        self.labels: Dict[int, List[NoduleLabel]] = {}  # slice_idx -> list of labels
        self.current_label_type = "nodule"
        self.current_series_path: Optional[Path] = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        # Left panel - Controls
        left_panel = QVBoxLayout()

        # Load button
        self.load_btn = QPushButton("Load DICOM Series")
        self.load_btn.clicked.connect(self.load_dicom_series)
        left_panel.addWidget(self.load_btn)

        # Series info
        self.info_label = QLabel("No series loaded")
        left_panel.addWidget(self.info_label)

        # Slice slider
        slider_layout = QVBoxLayout()
        self.slice_label = QLabel("Slice: 0/0")
        slider_layout.addWidget(self.slice_label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slider_layout.addWidget(self.slice_slider)
        left_panel.addLayout(slider_layout)

        # Label type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Label Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["nodule", "non-nodule", "suspicious"])
        self.type_combo.currentTextChanged.connect(self.on_label_type_changed)
        type_layout.addWidget(self.type_combo)
        left_panel.addLayout(type_layout)

        # Drawing tool selector
        tool_label = QLabel("Drawing Tool:")
        tool_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_panel.addWidget(tool_label)
        
        tool_buttons_layout = QVBoxLayout()
        
        # Create button group for exclusive selection
        from PyQt5.QtWidgets import QButtonGroup, QRadioButton
        self.tool_button_group = QButtonGroup()
        
        # Rectangle tool
        self.rect_btn = QRadioButton("🔲 Rectangle (Box)")
        self.rect_btn.setChecked(True)
        self.rect_btn.toggled.connect(lambda: self.set_draw_mode("rectangle"))
        self.tool_button_group.addButton(self.rect_btn)
        tool_buttons_layout.addWidget(self.rect_btn)
        
        # Circle tool
        self.circle_btn = QRadioButton("⭕ Circle")
        self.circle_btn.toggled.connect(lambda: self.set_draw_mode("circle"))
        self.tool_button_group.addButton(self.circle_btn)
        tool_buttons_layout.addWidget(self.circle_btn)
        
        # Ellipse tool
        self.ellipse_btn = QRadioButton("⬭ Ellipse (Oval)")
        self.ellipse_btn.toggled.connect(lambda: self.set_draw_mode("ellipse"))
        self.tool_button_group.addButton(self.ellipse_btn)
        tool_buttons_layout.addWidget(self.ellipse_btn)
        
        # Polygon tool
        self.polygon_btn = QRadioButton("📐 Polygon (Multi-point)")
        self.polygon_btn.toggled.connect(lambda: self.set_draw_mode("polygon"))
        self.tool_button_group.addButton(self.polygon_btn)
        tool_buttons_layout.addWidget(self.polygon_btn)
        
        # Freehand tool
        self.freehand_btn = QRadioButton("✏️ Freehand (Draw)")
        self.freehand_btn.toggled.connect(lambda: self.set_draw_mode("freehand"))
        self.tool_button_group.addButton(self.freehand_btn)
        tool_buttons_layout.addWidget(self.freehand_btn)
        
        left_panel.addLayout(tool_buttons_layout)
        
        # Tool instructions
        self.tool_instruction = QLabel("Click and drag to draw rectangle")
        self.tool_instruction.setWordWrap(True)
        self.tool_instruction.setStyleSheet("color: blue; font-size: 10px; margin: 5px;")
        left_panel.addWidget(self.tool_instruction)

        # Annotation list
        self.annotation_list = QListWidget()
        left_panel.addWidget(QLabel("Annotations on current slice:"))
        left_panel.addWidget(self.annotation_list)

        # Delete annotation button
        self.delete_btn = QPushButton("Delete Selected Annotation")
        self.delete_btn.clicked.connect(self.delete_annotation)
        left_panel.addWidget(self.delete_btn)

        # Clear slice button
        self.clear_btn = QPushButton("Clear Current Slice")
        self.clear_btn.clicked.connect(self.clear_slice)
        left_panel.addWidget(self.clear_btn)

        # Save button
        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.clicked.connect(self.save_annotations)
        left_panel.addWidget(self.save_btn)

        # Load annotations button
        self.load_ann_btn = QPushButton("Load Annotations")
        self.load_ann_btn.clicked.connect(self.load_annotations)
        left_panel.addWidget(self.load_ann_btn)

        # Statistics
        self.stats_label = QLabel("Total annotations: 0")
        left_panel.addWidget(self.stats_label)

        left_panel.addStretch()

        # Right panel - Image viewer
        right_panel = QVBoxLayout()
        self.image_viewer = ImageViewer()
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setMinimumSize(800, 800)
        right_panel.addWidget(self.image_viewer)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        central_widget.setLayout(main_layout)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_annotations)
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_dicom_series)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_annotation)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_slice)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_slice)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_annotation)

    def load_dicom_series(self):
        """Load a DICOM series from a directory"""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Series Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.current_series_path = folder_path
        self.dicom_files = sorted(list(folder_path.glob("*.dcm")))

        if not self.dicom_files:
            QMessageBox.warning(self, "Error", "No DICOM files found in the selected folder")
            return

        # Load volume
        try:
            self.volume = self.load_volume(self.dicom_files)
            self.labels = {}
            self.current_slice_idx = 0

            # Update UI
            self.slice_slider.setMaximum(len(self.dicom_files) - 1)
            self.slice_slider.setValue(0)
            self.info_label.setText(f"Loaded: {folder_path.name}\nSlices: {len(self.dicom_files)}")

            self.display_current_slice()
            self.update_stats()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM series: {str(e)}")

    def load_volume(self, dicom_files: List[Path]) -> np.ndarray:
        """Load DICOM volume"""
        slices = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(str(dcm_path))
            pixel_array = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu_image = pixel_array * slope + intercept
            slices.append(hu_image)

        volume = np.stack(slices)
        # Normalize
        volume = np.clip(volume, -1000, 400)
        volume = (volume - (-1000)) / (400 - (-1000))
        return volume

    def display_current_slice(self):
        """Display the current slice with annotations"""
        if self.volume is None:
            return

        slice_img = self.volume[self.current_slice_idx]
        self.image_viewer.set_image(slice_img)

        # Get annotations for current slice
        current_annotations = []
        if self.current_slice_idx in self.labels:
            current_annotations = [self._label_to_annotation(label) for label in self.labels[self.current_slice_idx]]

        self.image_viewer.set_annotations(current_annotations)
        self.slice_label.setText(f"Slice: {self.current_slice_idx + 1}/{len(self.dicom_files)}")
        self.update_annotation_list()

    def _label_to_annotation(self, label: NoduleLabel) -> Dict:
        """Convert NoduleLabel to annotation dict for display"""
        return {
            "shape": label.shape if hasattr(label, 'shape') else "rectangle",
            "bbox": label.bbox if hasattr(label, 'bbox') else None,
            "circle": label.circle if hasattr(label, 'circle') else None,
            "points": label.points if hasattr(label, 'points') else None,
        }

    def on_slice_changed(self, value: int):
        """Handle slice slider change"""
        # Save current slice annotations
        self.save_current_slice_annotations()

        self.current_slice_idx = value
        self.display_current_slice()

    def save_current_slice_annotations(self):
        """Save annotations from image viewer to labels dict"""
        if not self.image_viewer.annotations:
            # Remove empty slice from labels
            if self.current_slice_idx in self.labels:
                del self.labels[self.current_slice_idx]
            return

        # Convert viewer annotations to NoduleLabel objects
        labels = []
        for annotation in self.image_viewer.annotations:
            label = NoduleLabel(
                bbox=annotation.get("bbox"),
                slice_idx=self.current_slice_idx,
                label_type=self.current_label_type
            )
            # Add shape-specific data
            label.shape = annotation.get("shape", "rectangle")
            if "circle" in annotation:
                label.circle = annotation["circle"]
            if "points" in annotation:
                label.points = annotation["points"]
            labels.append(label)

        self.labels[self.current_slice_idx] = labels
        self.update_stats()

    def set_draw_mode(self, mode: str):
        """Set the drawing mode for the image viewer"""
        self.image_viewer.set_draw_mode(mode)
        
        # Update instruction text
        instructions = {
            "rectangle": "Click and drag to draw rectangle",
            "circle": "Click center, drag to set radius",
            "ellipse": "Click and drag to draw ellipse",
            "polygon": "Click to add points, double-click or right-click to finish",
            "freehand": "Click and drag to draw freely"
        }
        self.tool_instruction.setText(instructions.get(mode, ""))

    def on_label_type_changed(self, label_type: str):
        """Handle label type change"""
        self.current_label_type = label_type

    def update_annotation_list(self):
        """Update the annotation list widget"""
        self.annotation_list.clear()
        if self.current_slice_idx in self.labels:
            for i, label in enumerate(self.labels[self.current_slice_idx]):
                shape = getattr(label, 'shape', 'rectangle')
                
                if shape == "rectangle" and label.bbox:
                    x, y, w, h = label.bbox
                    desc = f"Box: ({x},{y}) {w}x{h}"
                elif shape == "circle" and hasattr(label, 'circle'):
                    x, y, r = label.circle
                    desc = f"Circle: center({x},{y}) r={r}"
                elif shape == "ellipse" and label.bbox:
                    x, y, w, h = label.bbox
                    desc = f"Ellipse: ({x},{y}) {w}x{h}"
                elif shape in ["polygon", "freehand"] and hasattr(label, 'points'):
                    desc = f"{shape.capitalize()}: {len(label.points)} points"
                else:
                    desc = "Unknown shape"
                
                self.annotation_list.addItem(f"{i + 1}. {label.label_type} - {desc}")

    def delete_annotation(self):
        """Delete selected annotation"""
        current_row = self.annotation_list.currentRow()
        if current_row >= 0 and self.current_slice_idx in self.labels:
            del self.labels[self.current_slice_idx][current_row]
            if not self.labels[self.current_slice_idx]:
                del self.labels[self.current_slice_idx]

            # Update viewer
            if self.current_slice_idx in self.labels:
                self.image_viewer.annotations = [self._label_to_annotation(label) for label in self.labels[self.current_slice_idx]]
            else:
                self.image_viewer.annotations = []

            self.image_viewer.update_display()
            self.update_annotation_list()
            self.update_stats()

    def clear_slice(self):
        """Clear all annotations on current slice"""
        if self.current_slice_idx in self.labels:
            del self.labels[self.current_slice_idx]
            self.image_viewer.annotations = []
            self.image_viewer.update_display()
            self.update_annotation_list()
            self.update_stats()

    def undo_last_annotation(self):
        """Undo the last annotation on current slice"""
        if self.image_viewer.annotations:
            self.image_viewer.annotations.pop()
            self.image_viewer.update_display()

    def prev_slice(self):
        """Go to previous slice"""
        if self.current_slice_idx > 0:
            self.slice_slider.setValue(self.current_slice_idx - 1)

    def next_slice(self):
        """Go to next slice"""
        if self.current_slice_idx < len(self.dicom_files) - 1:
            self.slice_slider.setValue(self.current_slice_idx + 1)

    def update_stats(self):
        """Update statistics display"""
        total = sum(len(labels) for labels in self.labels.values())
        slices_with_labels = len(self.labels)
        self.stats_label.setText(f"Total annotations: {total}\nSlices with labels: {slices_with_labels}")

    def save_annotations(self):
        """Save annotations to JSON file"""
        if not self.current_series_path:
            QMessageBox.warning(self, "Warning", "No series loaded")
            return

        # Save current slice first
        self.save_current_slice_annotations()

        default_name = self.current_series_path.name + "_annotations.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", str(self.current_series_path / default_name), "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            # Convert labels to serializable format
            data = {
                "series_path": str(self.current_series_path),
                "num_slices": len(self.dicom_files),
                "annotations": {str(k): [label.to_dict() for label in v] for k, v in self.labels.items()},
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            QMessageBox.information(self, "Success", f"Annotations saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {str(e)}")

    def load_annotations(self):
        """Load annotations from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json)")

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert back to NoduleLabel objects
            self.labels = {}
            for slice_idx_str, labels_data in data["annotations"].items():
                slice_idx = int(slice_idx_str)
                self.labels[slice_idx] = [NoduleLabel.from_dict(label_data) for label_data in labels_data]

            self.display_current_slice()
            self.update_stats()
            QMessageBox.information(self, "Success", "Annotations loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load annotations: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
