"""
Batch export utilities for preparing training datasets
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom


def load_dicom_volume(dicom_dir: Path) -> np.ndarray:
    """Load DICOM volume from directory"""
    dicom_files = sorted(list(dicom_dir.glob("*.dcm")))
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


def export_labeled_slices(annotations_json: Path, dicom_dir: Path, output_dir: Path) -> None:
    """
    Export labeled slices as PNG images with corresponding label files
    
    Args:
        annotations_json: Path to annotations JSON file
        dicom_dir: Path to DICOM series directory
        output_dir: Output directory for images and labels
    """
    # Load annotations
    with open(annotations_json, "r") as f:
        data = json.load(f)

    # Load volume
    volume = load_dicom_volume(dicom_dir)

    # Create output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Export slices with annotations
    class_mapping = {"nodule": 0, "suspicious": 1, "non-nodule": 2}

    for slice_idx_str, labels in data["annotations"].items():
        slice_idx = int(slice_idx_str)
        slice_img = volume[slice_idx]

        # Save image
        img_normalized = (slice_img * 255).astype(np.uint8)
        img_path = images_dir / f"slice_{slice_idx:04d}.png"

        import cv2

        cv2.imwrite(str(img_path), img_normalized)

        # Save labels in YOLO format
        label_path = labels_dir / f"slice_{slice_idx:04d}.txt"
        with open(label_path, "w") as f:
            for label in labels:
                bbox = label["bbox"]
                label_type = label.get("label_type", "nodule")
                class_id = class_mapping.get(label_type, 0)

                # YOLO format: class x_center y_center width height (normalized)
                h, w = slice_img.shape
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                width = bbox[2] / w
                height = bbox[3] / h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Exported {len(data['annotations'])} labeled slices to {output_dir}")


def create_dataset_yaml(output_dir: Path, dataset_name: str = "nodule_detection") -> None:
    """
    Create YAML configuration file for YOLO training
    
    Args:
        output_dir: Dataset root directory
        dataset_name: Name of the dataset
    """
    yaml_content = f"""# {dataset_name} Dataset Configuration

path: {output_dir.absolute()}  # dataset root dir
train: images  # train images directory
val: images    # val images directory (same as train for now)

# Classes
names:
  0: nodule
  1: suspicious
  2: non-nodule

# Number of classes
nc: 3
"""

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Created dataset configuration: {yaml_path}")


def split_dataset(
    images_dir: Path, labels_dir: Path, output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> None:
    """
    Split dataset into train/val/test sets
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    import shutil
    import random

    # Get all image files
    image_files = sorted(list(images_dir.glob("*.png")))
    random.shuffle(image_files)

    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train : n_train + n_val]
    test_files = image_files[n_train + n_val :]

    # Create directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_split(files: List[Path], split_name: str):
        for img_file in files:
            # Copy image
            shutil.copy(img_file, output_dir / split_name / "images" / img_file.name)

            # Copy label
            label_file = labels_dir / img_file.with_suffix(".txt").name
            if label_file.exists():
                shutil.copy(label_file, output_dir / split_name / "labels" / label_file.name)

    copy_split(train_files, "train")
    copy_split(val_files, "val")
    copy_split(test_files, "test")

    print(f"Dataset split complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch export labeled data")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations JSON")
    parser.add_argument("--dicom-dir", type=str, required=True, help="Path to DICOM series directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test")

    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    dicom_dir = Path(args.dicom_dir)
    output_dir = Path(args.output)

    # Export labeled slices
    temp_dir = output_dir / "temp"
    export_labeled_slices(annotations_path, dicom_dir, temp_dir)

    if args.split:
        # Split dataset
        split_dataset(temp_dir / "images", temp_dir / "labels", output_dir)
        create_dataset_yaml(output_dir)
    else:
        # Just create YAML for the exported data
        create_dataset_yaml(temp_dir)

    print("Export complete!")
