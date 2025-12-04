from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import amp

from .annotations import NoduleAnnotation

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - optional dependency
    ndimage = None  # type: ignore


def infer_nodules(
    model: torch.nn.Module,
    volume: torch.Tensor,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        device_type = volume.device.type if hasattr(volume, "device") else "cpu"
        with amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(volume.unsqueeze(0))
        probs = torch.sigmoid(logits)[0, 0]
    binary = (probs >= threshold).cpu().numpy().astype(np.uint8)
    return binary


def estimate_lung_mask(
    volume: np.ndarray,
    lung_threshold_hu: int = -320,
    min_area: int = 1500,
) -> Optional[np.ndarray]:
    if ndimage is None:
        return None
    if volume.ndim != 3:
        return None

    min_hu, max_hu = -1000.0, 400.0
    hu_volume = np.clip(volume, 0.0, 1.0) * (max_hu - min_hu) + min_hu
    threshold = lung_threshold_hu
    air_mask = hu_volume <= threshold
    lung_mask = np.zeros_like(air_mask, dtype=bool)
    structure2d = ndimage.generate_binary_structure(rank=2, connectivity=1)
    border_template = np.zeros(air_mask.shape[1:], dtype=bool)
    border_template[0, :] = True
    border_template[-1, :] = True
    border_template[:, 0] = True
    border_template[:, -1] = True

    for idx in range(air_mask.shape[0]):
        slice_mask = air_mask[idx]
        if not slice_mask.any():
            continue
        labeled, num_features = ndimage.label(slice_mask, structure=structure2d)
        if num_features == 0:
            continue
        slice_clean = slice_mask.copy()
        border_labels = np.unique(labeled[border_template])
        for label_id in border_labels:
            if label_id == 0:
                continue
            slice_clean[labeled == label_id] = False
        if slice_clean.sum() < min_area:
            continue
        labeled, num_features = ndimage.label(slice_clean, structure=structure2d)
        if num_features == 0:
            continue
        areas = ndimage.sum(slice_clean, labeled, index=range(1, num_features + 1))
        label_area_pairs = sorted(
            ((label_id, area) for label_id, area in enumerate(areas, start=1)),
            key=lambda item: item[1],
            reverse=True,
        )
        keep_labels = [label for label, _ in label_area_pairs[:2]]
        slice_keep = np.isin(labeled, keep_labels)
        slice_keep = ndimage.binary_closing(slice_keep, structure=np.ones((5, 5)), iterations=1)
        slice_keep = ndimage.binary_fill_holes(slice_keep)
        if slice_keep.sum() < min_area:
            continue
        lung_mask[idx] = slice_keep

    if not lung_mask.any():
        return None

    structure3d = ndimage.generate_binary_structure(rank=3, connectivity=1)
    lung_mask = ndimage.binary_closing(lung_mask, structure=structure3d, iterations=1)
    return lung_mask.astype(np.uint8)


def postprocess_nodules(
    mask: np.ndarray,
    min_voxels: int = 10,
    min_slices: int = 1,
    return_labels: bool = False,
) -> List[Dict[str, float]] | Tuple[List[Dict[str, float]], np.ndarray]:
    if ndimage is None:
        raise ImportError("scipy is required for post-processing connected components")
    if min_slices < 1:
        raise ValueError("min_slices must be >= 1")

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    needs_cleanup = min_voxels > 1 or min_slices > 1
    cleaned_mask = (
        ndimage.binary_opening(mask.astype(bool), structure=structure)
        if needs_cleanup
        else mask.astype(bool)
    )
    labeled, num_features = ndimage.label(cleaned_mask)
    nodules: List[Dict[str, float]] = []
    for idx in range(1, num_features + 1):
        component = (labeled == idx)
        voxel_count = int(component.sum())
        if voxel_count < min_voxels:
            continue
        slice_profile = component.any(axis=(1, 2))
        slice_count = int(slice_profile.sum())
        if slice_count < min_slices:
            continue
        centroid = ndimage.center_of_mass(component)
        nodules.append(
            {
                "id": idx,
                "voxel_count": voxel_count,
                "slice_count": slice_count,
                "centroid_z": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_x": float(centroid[2]),
            }
        )
    if return_labels:
        return nodules, labeled
    return nodules


def analyze_nodules(
    nodules: List[Dict[str, float]],
    annotations: Sequence[NoduleAnnotation],
    sop_uid_to_index: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    report: List[Dict[str, object]] = []
    for nodule in nodules:
        closest_annotation: Optional[NoduleAnnotation] = None
        min_distance = float("inf")
        for annotation in annotations:
            centroid = annotation.centroid(sop_uid_to_index)
            if centroid is None:
                continue
            distance = math.sqrt(
                (centroid[0] - nodule["centroid_z"]) ** 2
                + (centroid[1] - nodule["centroid_y"]) ** 2
                + (centroid[2] - nodule["centroid_x"]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_annotation = annotation
        characteristics = closest_annotation.characteristics if closest_annotation else {}
        report.append(
            {
                "detected_id": nodule["id"],
                "voxel_count": nodule["voxel_count"],
                "centroid": [
                    nodule["centroid_z"],
                    nodule["centroid_y"],
                    nodule["centroid_x"],
                ],
                "matched_annotation": closest_annotation.nodule_id if closest_annotation else None,
                "malignancy_score": characteristics.get("malignancy"),
                "characteristics": characteristics,
                "distance_to_annotation": min_distance if closest_annotation else None,
            }
        )
    return report


def extract_filtered_mask(
    mask: np.ndarray,
    nodules: Sequence[Dict[str, float]],
    labeled_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if ndimage is None:
        return mask.astype(np.uint8), None
    if not nodules:
        return np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.int32)
    if labeled_mask is None:
        labeled_mask, _ = ndimage.label(mask)
    keep_ids = {int(nodule.get("id", 0)) for nodule in nodules if nodule.get("id") is not None}
    if not keep_ids:
        return np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.int32)
    keep_mask = np.isin(labeled_mask, list(keep_ids))
    filtered = keep_mask.astype(np.uint8)
    labeled_filtered = np.where(keep_mask, labeled_mask, 0).astype(np.int32)
    return filtered, labeled_filtered


__all__ = [
    "infer_nodules",
    "postprocess_nodules",
    "analyze_nodules",
    "extract_filtered_mask",
    "estimate_lung_mask",
]
