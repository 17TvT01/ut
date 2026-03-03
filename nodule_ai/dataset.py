from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .annotations import NoduleAnnotation, parse_annotation_xml
from .dicom import build_nodule_mask, load_dicom_series

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency at import time
    sitk = None  # type: ignore


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
        volume_np, meta, _ = load_dicom_series(series_dir)
        mask_np = build_nodule_mask(volume_np.shape, sample["annotations"], meta)
        volume = torch.from_numpy(volume_np).unsqueeze(0)  # (1, Z, Y, X)
        mask = torch.from_numpy(mask_np.astype("float32")).unsqueeze(0)
        if self.target_shape is not None:
            volume, mask = self._downsample_pair(volume, mask, self.target_shape)
        batch: Dict[str, object] = {
            "volume": volume,
            "mask": mask,
            "sample_id": str(Path(series_dir).name),
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
        if tensor.shape[1:] == target:
            return tensor
        kwargs: Dict[str, bool] = {}
        if mode in {"trilinear", "bilinear"}:
            kwargs["align_corners"] = False
        tensor = tensor.to(dtype=torch.float32)
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


class LUNA16Dataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        cache: bool = False,
        target_shape: Tuple[int, int, int] | None = None,
        transform=None,
        annotations_csv: Path | str | None = None,
        include_negative: bool = True,
    ) -> None:
        if sitk is None:
            raise ImportError("SimpleITK is required for LUNA16. Please install SimpleITK.")

        self.root = Path(root)
        self.cache = cache
        self.target_shape = target_shape
        self.transform = transform

        annotations_path = self._resolve_annotations_csv(annotations_csv)
        self._annotations_map = self._load_annotations(annotations_path)

        self.samples: List[Dict[str, object]] = []
        for mhd_path in sorted(self.root.rglob("*.mhd")):
            series_uid = mhd_path.stem
            series_annotations = self._annotations_map.get(series_uid, [])
            if series_annotations or include_negative:
                self.samples.append(
                    {
                        "series_uid": series_uid,
                        "mhd_path": mhd_path,
                        "annotations": series_annotations,
                    }
                )

        if not self.samples:
            raise ValueError(f"No valid LUNA16 studies found under {self.root}")

        self._cache: Dict[int, Dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_annotations_csv(self, annotations_csv: Path | str | None) -> Optional[Path]:
        if annotations_csv is not None:
            candidate = Path(annotations_csv)
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"LUNA16 annotations.csv not found: {candidate}")

        candidates = [
            self.root / "annotations.csv",
            self.root / "CSVFILES" / "annotations.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_annotations(self, annotations_path: Optional[Path]) -> Dict[str, List[Dict[str, float]]]:
        annotations_map: Dict[str, List[Dict[str, float]]] = {}
        if annotations_path is None:
            return annotations_map

        with annotations_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_cols = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
            if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Invalid LUNA16 annotations file: {annotations_path}. "
                    f"Required columns: {sorted(required_cols)}"
                )

            for row in reader:
                try:
                    series_uid = str(row["seriesuid"]).strip()
                    ann = {
                        "coordX": float(row["coordX"]),
                        "coordY": float(row["coordY"]),
                        "coordZ": float(row["coordZ"]),
                        "diameter_mm": float(row["diameter_mm"]),
                    }
                except (TypeError, ValueError, KeyError):
                    continue
                if not series_uid:
                    continue
                annotations_map.setdefault(series_uid, []).append(ann)

        return annotations_map

    @staticmethod
    def _normalize_hounsfield(volume: np.ndarray, clip: Tuple[int, int] = (-1000, 400)) -> np.ndarray:
        min_hu, max_hu = clip
        volume = np.clip(volume, min_hu, max_hu)
        volume = (volume - min_hu) / float(max_hu - min_hu)
        return volume.astype(np.float32)

    @staticmethod
    def _build_luna_mask(
        shape_zyx: Tuple[int, int, int],
        annotations: List[Dict[str, float]],
        origin_xyz: np.ndarray,
        spacing_xyz: np.ndarray,
        direction_matrix: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros(shape_zyx, dtype=np.uint8)
        if not annotations:
            return mask

        inv_direction = np.linalg.inv(direction_matrix)
        depth, height, width = shape_zyx

        for ann in annotations:
            world_xyz = np.array([ann["coordX"], ann["coordY"], ann["coordZ"]], dtype=np.float64)
            diameter_mm = max(float(ann.get("diameter_mm", 0.0)), 0.0)
            radius_mm = max(diameter_mm / 2.0, 1.0)

            voxel_xyz = inv_direction.dot(world_xyz - origin_xyz) / spacing_xyz
            center_x, center_y, center_z = voxel_xyz.tolist()

            radius_x = max(radius_mm / max(spacing_xyz[0], 1e-6), 1.0)
            radius_y = max(radius_mm / max(spacing_xyz[1], 1e-6), 1.0)
            radius_z = max(radius_mm / max(spacing_xyz[2], 1e-6), 1.0)

            x0 = max(0, int(np.floor(center_x - radius_x - 1)))
            x1 = min(width - 1, int(np.ceil(center_x + radius_x + 1)))
            y0 = max(0, int(np.floor(center_y - radius_y - 1)))
            y1 = min(height - 1, int(np.ceil(center_y + radius_y + 1)))
            z0 = max(0, int(np.floor(center_z - radius_z - 1)))
            z1 = min(depth - 1, int(np.ceil(center_z + radius_z + 1)))

            if x0 > x1 or y0 > y1 or z0 > z1:
                continue

            zz, yy, xx = np.ogrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            ellipsoid = (
                ((xx - center_x) / radius_x) ** 2
                + ((yy - center_y) / radius_y) ** 2
                + ((zz - center_z) / radius_z) ** 2
            ) <= 1.0
            mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1][ellipsoid] = 1

        return mask

    def _load_item(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        series_uid = str(sample["series_uid"])
        mhd_path = Path(sample["mhd_path"])
        annotations = sample["annotations"]

        image = sitk.ReadImage(str(mhd_path))
        volume_np = sitk.GetArrayFromImage(image).astype(np.float32)  # Z, Y, X
        volume_np = self._normalize_hounsfield(volume_np)

        origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
        spacing_xyz = np.array(image.GetSpacing(), dtype=np.float64)
        direction_matrix = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

        mask_np = self._build_luna_mask(
            volume_np.shape,
            annotations if isinstance(annotations, list) else [],
            origin_xyz,
            spacing_xyz,
            direction_matrix,
        )

        volume = torch.from_numpy(volume_np).unsqueeze(0)
        mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
        if self.target_shape is not None:
            volume, mask = LIDCDataset._downsample_pair(volume, mask, self.target_shape)

        batch: Dict[str, object] = {
            "volume": volume,
            "mask": mask,
            "sample_id": series_uid,
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


def build_dataset(
    root: Path | str,
    cache: bool = False,
    target_shape: Tuple[int, int, int] | None = None,
    transform=None,
) -> Dataset:
    root_path = Path(root)
    has_mhd = any(root_path.rglob("*.mhd"))
    has_xml = any(root_path.rglob("*.xml"))

    if has_mhd and not has_xml:
        return LUNA16Dataset(root_path, cache=cache, target_shape=target_shape, transform=transform)

    if has_xml:
        return LIDCDataset(root_path, cache=cache, target_shape=target_shape, transform=transform)

    if has_mhd:
        return LUNA16Dataset(root_path, cache=cache, target_shape=target_shape, transform=transform)

    raise ValueError(
        f"Cannot determine dataset format under {root_path}. "
        "Expected LIDC (XML + DICOM) or LUNA16 (*.mhd + annotations.csv)."
    )


__all__ = ["LIDCDataset", "LUNA16Dataset", "build_dataset"]
