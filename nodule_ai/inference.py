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


def postprocess_nodules(mask: np.ndarray, min_voxels: int = 10) -> List[Dict[str, float]]:
    if ndimage is None:
        raise ImportError("scipy is required for post-processing connected components")
    labeled, num_features = ndimage.label(mask)
    nodules: List[Dict[str, float]] = []
    for idx in range(1, num_features + 1):
        component = (labeled == idx)
        voxel_count = int(component.sum())
        if voxel_count < min_voxels:
            continue
        centroid = ndimage.center_of_mass(component)
        nodules.append(
            {
                "id": idx,
                "voxel_count": voxel_count,
                "centroid_z": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_x": float(centroid[2]),
            }
        )
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


def extract_filtered_mask(mask: np.ndarray, nodules: Sequence[Dict[str, float]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if ndimage is None:
        return mask.astype(np.uint8), None
    if not nodules:
        return np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.int32)
    labeled, _ = ndimage.label(mask)
    keep_ids = {int(nodule.get("id", 0)) for nodule in nodules if nodule.get("id") is not None}
    if not keep_ids:
        return np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.int32)
    keep_mask = np.isin(labeled, list(keep_ids))
    filtered = keep_mask.astype(np.uint8)
    labeled_filtered = np.where(keep_mask, labeled, 0).astype(np.int32)
    return filtered, labeled_filtered


__all__ = ["infer_nodules", "postprocess_nodules", "analyze_nodules", "extract_filtered_mask"]
