from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .annotations import NoduleAnnotation, parse_annotation_xml
from .dicom import build_nodule_mask, load_dicom_series


class LIDCDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        cache: bool = False,
        target_shape: Tuple[int, int, int] | None = None,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.cache = cache
        self.target_shape = target_shape
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
        if self.target_shape is not None:
            volume, mask = self._downsample_pair(volume, mask, self.target_shape)
        batch: Dict[str, object] = {
            "volume": volume,
            "mask": mask,
        }
        if self.transform:
            batch = self.transform(batch)
        return batch

    @staticmethod
    def _center_crop(tensor: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        _, depth, height, width = tensor.shape
        target_depth, target_height, target_width = target_shape
        depth_slice = LIDCDataset._compute_slice(depth, target_depth)
        height_slice = LIDCDataset._compute_slice(height, target_height)
        width_slice = LIDCDataset._compute_slice(width, target_width)
        return tensor[:, depth_slice, height_slice, width_slice]

    @staticmethod
    def _compute_slice(size: int, target: int) -> slice:
        if size <= target:
            return slice(0, size)
        start = max((size - target) // 2, 0)
        end = start + target
        return slice(start, min(end, size))

    @staticmethod
    def _resize_tensor(
        tensor: torch.Tensor,
        target_shape: Tuple[int, int, int],
        mode: str,
    ) -> torch.Tensor:
        target = tuple(int(dim) for dim in target_shape)
        current_shape = tensor.shape[1:]
        if any(curr > tgt for curr, tgt in zip(current_shape, target)):
            tensor = LIDCDataset._center_crop(tensor, target)
            current_shape = tensor.shape[1:]
        if current_shape == target:
            return tensor
        kwargs: Dict[str, bool] = {}
        if mode in {"trilinear", "bilinear"}:
            kwargs["align_corners"] = False
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=target,
            mode=mode,
            **kwargs,
        ).squeeze(0)
        return tensor

    @staticmethod
    def _downsample_pair(
        volume: torch.Tensor,
        mask: torch.Tensor,
        target_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = LIDCDataset._resize_tensor(volume, target_shape, mode="trilinear")
        mask = LIDCDataset._resize_tensor(mask, target_shape, mode="nearest")
        return volume, mask

    @staticmethod
    def downsample_volume(volume: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        return LIDCDataset._resize_tensor(volume, target_shape, mode="trilinear")

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        item = self._load_item(idx)
        if self.cache:
            self._cache[idx] = item
        return item


__all__ = ["LIDCDataset"]
