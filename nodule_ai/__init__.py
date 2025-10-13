"""Core package for lung nodule detection workflow."""

from .annotations import (
    EdgeCoord,
    NoduleSlice,
    NoduleAnnotation,
    parse_annotation_xml,
)
from .dicom import normalize_hounsfield, load_dicom_series, build_nodule_mask
from .dataset import LIDCDataset
from .model import ComplexUNet3D, UNet3D
from .training import dice_loss, train_epoch, evaluate_epoch
from .trainer import (
    TrainingConfig,
    build_dataloaders,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from .inference import infer_nodules, postprocess_nodules, analyze_nodules
from .report import generate_report, save_report

__all__ = [
    "EdgeCoord",
    "NoduleSlice",
    "NoduleAnnotation",
    "parse_annotation_xml",
    "normalize_hounsfield",
    "load_dicom_series",
    "build_nodule_mask",
    "LIDCDataset",
    "UNet3D",
    "ComplexUNet3D",
    "dice_loss",
    "train_epoch",
    "evaluate_epoch",
    "TrainingConfig",
    "build_dataloaders",
    "load_checkpoint",
    "save_checkpoint",
    "train_model",
    "infer_nodules",
    "postprocess_nodules",
    "analyze_nodules",
    "generate_report",
    "save_report",
]
