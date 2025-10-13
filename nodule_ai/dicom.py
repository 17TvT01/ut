from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pydicom

from .annotations import NoduleAnnotation


def normalize_hounsfield(volume: np.ndarray, clip: Tuple[int, int] = (-1000, 400)) -> np.ndarray:
    min_hu, max_hu = clip
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / float(max_hu - min_hu)
    return volume.astype(np.float32)


def load_dicom_series(series_dir: Path) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    dicom_files = sorted(series_dir.rglob("*.dcm"))
    slice_records = []
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(str(dcm_path))
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu_image = pixel_array * slope + intercept
        position = getattr(ds, "ImagePositionPatient", None)
        instance_number = getattr(ds, "InstanceNumber", None)
        if position is not None:
            z_pos = float(position[2])
        else:
            z_pos = float(instance_number or len(slice_records))
        meta = {
            "path": str(dcm_path),
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "instance_number": float(instance_number or len(slice_records)),
            "z_position": z_pos,
        }
        slice_records.append((z_pos, hu_image, meta))
    if not slice_records:
        raise ValueError(f"No DICOM files found in {series_dir}")
    slice_records.sort(key=lambda record: record[0])
    volume = np.stack([record[1] for record in slice_records])
    volume = normalize_hounsfield(volume)
    meta_info = {
        record[2]["sop_uid"]: {
            "index": idx,
            "z_position": record[0],
            "path": record[2]["path"],
        }
        for idx, record in enumerate(slice_records)
    }
    return volume, meta_info


def build_nodule_mask(
    volume_shape: Tuple[int, int, int],
    annotations: Sequence[NoduleAnnotation],
    sop_uid_to_index: Dict[str, Dict[str, float]],
    dilation: int = 2,
) -> np.ndarray:
    mask = np.zeros(volume_shape, dtype=np.uint8)
    for nodule in annotations:
        for slice_annotation in nodule.slices:
            meta = sop_uid_to_index.get(slice_annotation.sop_uid)
            if meta is None or not slice_annotation.edges:
                continue
            z_idx = int(meta["index"])
            xs = [edge.x for edge in slice_annotation.edges]
            ys = [edge.y for edge in slice_annotation.edges]
            x0, x1 = max(min(xs) - dilation, 0), min(max(xs) + dilation, volume_shape[2] - 1)
            y0, y1 = max(min(ys) - dilation, 0), min(max(ys) + dilation, volume_shape[1] - 1)
            mask[z_idx, y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


__all__ = [
    "normalize_hounsfield",
    "load_dicom_series",
    "build_nodule_mask",
]
