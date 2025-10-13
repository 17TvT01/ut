from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from .dataset import LIDCDataset
from .inference import analyze_nodules, infer_nodules, postprocess_nodules


def generate_report(
    model: torch.nn.Module,
    dataset: LIDCDataset,
    study_index: int,
    threshold: float = 0.5,
) -> Dict[str, object]:
    batch = dataset[study_index]
    volume = batch["volume"].to(dtype=torch.float32)
    binary_mask = infer_nodules(model, volume, threshold=threshold)
    nodules = postprocess_nodules(binary_mask)
    detailed = analyze_nodules(nodules, batch["annotations"], batch["meta"])
    return {
        "study": str(dataset.samples[study_index]["study_dir"]),
        "nodule_count": len(detailed),
        "nodules": detailed,
    }


def save_report(report: Dict[str, object], output_path: Path | str) -> None:
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


__all__ = ["generate_report", "save_report"]
