"""
Utilities to convert labeled data to training format
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_annotations(json_path: Path) -> Dict:
    """Load annotations from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


def convert_to_yolo_format(
    annotations_json: Path, output_dir: Path, image_width: int = 512, image_height: int = 512
) -> None:
    """
    Convert annotations to YOLO format (class x_center y_center width height)
    All values are normalized to [0, 1]
    """
    data = load_annotations(annotations_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_mapping = {"nodule": 0, "suspicious": 1, "non-nodule": 2}

    for slice_idx_str, labels in data["annotations"].items():
        slice_idx = int(slice_idx_str)
        output_file = output_dir / f"slice_{slice_idx:04d}.txt"

        with open(output_file, "w") as f:
            for label in labels:
                bbox = label["bbox"]  # [x, y, w, h]
                label_type = label.get("label_type", "nodule")
                class_id = class_mapping.get(label_type, 0)

                # Convert to YOLO format (normalized)
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"YOLO format labels saved to {output_dir}")


def convert_to_coco_format(annotations_json: Path, output_json: Path) -> None:
    """Convert annotations to COCO format"""
    data = load_annotations(annotations_json)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "nodule"},
            {"id": 1, "name": "suspicious"},
            {"id": 2, "name": "non-nodule"},
        ],
    }

    class_mapping = {"nodule": 0, "suspicious": 1, "non-nodule": 2}
    annotation_id = 0

    for slice_idx_str, labels in data["annotations"].items():
        slice_idx = int(slice_idx_str)
        image_id = slice_idx

        # Add image info
        coco_format["images"].append(
            {
                "id": image_id,
                "file_name": f"slice_{slice_idx:04d}.png",
                "width": 512,
                "height": 512,
            }
        )

        # Add annotations
        for label in labels:
            bbox = label["bbox"]  # [x, y, w, h]
            label_type = label.get("label_type", "nodule")
            category_id = class_mapping.get(label_type, 0)

            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(f"COCO format saved to {output_json}")


def extract_patches(
    annotations_json: Path,
    volume: np.ndarray,
    output_dir: Path,
    patch_size: Tuple[int, int] = (64, 64),
    save_negative: bool = True,
) -> None:
    """
    Extract image patches around annotations for training
    
    Args:
        annotations_json: Path to annotations JSON
        volume: 3D numpy array of CT volume
        output_dir: Directory to save patches
        patch_size: Size of patches to extract (height, width)
        save_negative: Whether to save negative (non-nodule) patches
    """
    data = load_annotations(annotations_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_dir = output_dir / "positive"
    negative_dir = output_dir / "negative"
    suspicious_dir = output_dir / "suspicious"

    positive_dir.mkdir(exist_ok=True)
    negative_dir.mkdir(exist_ok=True)
    suspicious_dir.mkdir(exist_ok=True)

    patch_h, patch_w = patch_size

    for slice_idx_str, labels in data["annotations"].items():
        slice_idx = int(slice_idx_str)
        slice_img = volume[slice_idx]

        for i, label in enumerate(labels):
            bbox = label["bbox"]  # [x, y, w, h]
            label_type = label.get("label_type", "nodule")

            # Calculate patch center
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)

            # Extract patch
            y1 = max(0, center_y - patch_h // 2)
            y2 = min(slice_img.shape[0], center_y + patch_h // 2)
            x1 = max(0, center_x - patch_w // 2)
            x2 = min(slice_img.shape[1], center_x + patch_w // 2)

            patch = slice_img[y1:y2, x1:x2]

            # Pad if necessary
            if patch.shape != patch_size:
                padded = np.zeros(patch_size, dtype=patch.dtype)
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded

            # Save patch
            if label_type == "nodule":
                save_dir = positive_dir
            elif label_type == "suspicious":
                save_dir = suspicious_dir
            else:
                save_dir = negative_dir

            filename = f"slice_{slice_idx:04d}_bbox_{i:02d}.npy"
            np.save(save_dir / filename, patch)

    print(f"Patches extracted to {output_dir}")


def generate_training_dataset(
    annotations_json: Path, volume: np.ndarray, output_dir: Path, format: str = "yolo"
) -> None:
    """
    Generate complete training dataset from annotations
    
    Args:
        annotations_json: Path to annotations JSON
        volume: 3D numpy array of CT volume
        output_dir: Directory to save dataset
        format: Output format ('yolo', 'coco', or 'patches')
    """
    if format == "yolo":
        labels_dir = output_dir / "labels"
        convert_to_yolo_format(annotations_json, labels_dir)
    elif format == "coco":
        coco_json = output_dir / "annotations.json"
        convert_to_coco_format(annotations_json, coco_json)
    elif format == "patches":
        extract_patches(annotations_json, volume, output_dir)
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert labeled data to training format")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--format", type=str, choices=["yolo", "coco", "patches"], default="yolo", help="Output format"
    )

    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    output_path = Path(args.output)

    if args.format in ["yolo", "coco"]:
        if args.format == "yolo":
            convert_to_yolo_format(annotations_path, output_path / "labels")
        else:
            convert_to_coco_format(annotations_path, output_path / "annotations.json")
    else:
        print("For patches format, you need to provide the volume separately")
