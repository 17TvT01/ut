from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .annotations import NoduleAnnotation, parse_annotation_xml
from .dicom import build_nodule_mask, load_dicom_series


class LIDCDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        cache: bool = False,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.cache = cache
        self.samples: List[Dict[str, object]] = []
        for xml_path in sorted(self.root.rglob("*.xml")):
            series_dir = xml_path.parent
            if not series_dir.is_dir():
                continue
            try:
                annotations = parse_annotation_xml(xml_path)
            except Exception:
                # Skip malformed annotation files to keep dataset construction resilient
                continue
            has_dicoms = any(series_dir.glob("*.dcm")) or any(series_dir.glob("*.DCM"))
            if not has_dicoms:
                has_dicoms = any(series_dir.glob("**/*.dcm")) or any(series_dir.glob("**/*.DCM"))
            if not has_dicoms:
                continue
            self.samples.append({
                "series_dir": series_dir,
                "xml_path": xml_path,
                "annotations": annotations,
            })
        if not self.samples:
            raise ValueError(f"No valid studies found under {self.root}")
        self._cache: Dict[int, Dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_item(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        series_dir = sample["series_dir"]
        volume_np, meta = load_dicom_series(series_dir)
        mask_np = build_nodule_mask(volume_np.shape, sample["annotations"], meta)
        volume = torch.from_numpy(volume_np).unsqueeze(0)  # (1, Z, Y, X)
        mask = torch.from_numpy(mask_np.astype("float32")).unsqueeze(0)
        batch: Dict[str, object] = {
            "volume": volume,
            "mask": mask,
        }
        if self.transform:
            batch = self.transform(batch)
        return batch

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        item = self._load_item(idx)
        if self.cache:
            self._cache[idx] = item
        return item


__all__ = ["LIDCDataset"]
