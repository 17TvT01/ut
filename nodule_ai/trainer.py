from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torch import amp

from .dataset import LIDCDataset
from .model import ComplexUNet3D
from .training import evaluate_epoch, train_epoch


@dataclass
class TrainingConfig:
    data_dir: Path
    epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-3
    base_filters: int = 32
    dropout: float = 0.1
    upsample_mode: str = "trilinear"
    n_channels: int = 1
    n_classes: int = 1
    pin_memory: bool | None = None
    use_amp: bool | None = None
    val_split: float = 0.2
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    checkpoint: Path = Path("checkpoints/complex_unet3d.pt")
    resume: Path | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def export_training_history(history: Dict[str, list[float]], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history:
        return
    csv_path = json_path.with_suffix(".csv")
    columns = ["epoch", *history.keys()]
    epochs = len(next(iter(history.values()), []))
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for idx in range(epochs):
            row = [idx + 1]
            for key in history:
                row.append(history[key][idx])
            writer.writerow(row)


def export_training_summary(summary: Dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    dataset = LIDCDataset(data_dir, cache=False)
    if len(dataset) < 2:
        raise ValueError("Need at least 2 studies to create validation split")
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size == 0:
        raise ValueError("Validation split too large; no samples left for training")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint.get("epoch", 0))


def train_model(
    config: TrainingConfig,
    progress: Optional[Callable[[int, int, str, Optional[float], Optional[float], Optional[str]], None]] = None,
) -> Dict[str, list[float]]:
    set_seed(config.seed)
    device = torch.device(config.device)
    auto_pin_memory = config.pin_memory if config.pin_memory is not None else device.type == "cuda"
    use_amp = config.use_amp if config.use_amp is not None else device.type == "cuda"
    non_blocking = auto_pin_memory and device.type == "cuda"
    train_loader, val_loader = build_dataloaders(
        config.data_dir,
        config.batch_size,
        config.val_split,
        config.num_workers,
        config.seed,
        pin_memory=auto_pin_memory,
    )
    model = ComplexUNet3D(
        n_channels=config.n_channels,
        n_classes=config.n_classes,
        base_filters=config.base_filters,
        dropout=config.dropout,
        upsample_mode=config.upsample_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    if config.resume and config.resume.exists():
        start_epoch = load_checkpoint(model, optimizer, config.resume) + 1

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = start_epoch - 1

    for epoch in range(start_epoch, config.epochs + 1):
        if progress:
            progress(epoch, config.epochs, "message", None, None, f"Epoch {epoch}/{config.epochs}: bat dau huan luyen")
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
            non_blocking=non_blocking,
        )
        if progress:
            progress(epoch, config.epochs, "message", None, None, "Dang danh gia tren tap validation")
        val_loss = evaluate_epoch(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            non_blocking=non_blocking,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if progress:
            progress(epoch, config.epochs, "metrics", train_loss, val_loss, f"Epoch {epoch} hoan tat: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, config.checkpoint)
            if progress:
                progress(epoch, config.epochs, "message", None, None, f"Checkpoint moi luu (val={val_loss:.4f})")

    history_path = config.checkpoint.with_suffix(".history.json")
    export_training_history(history, history_path)
    summary = {
        "model": "ComplexUNet3D",
        "model_params": {
            "n_channels": config.n_channels,
            "n_classes": config.n_classes,
            "base_filters": config.base_filters,
            "dropout": config.dropout,
            "upsample_mode": config.upsample_mode,
        },
        "use_amp": use_amp,
        "pin_memory": auto_pin_memory,
        "non_blocking_transfers": non_blocking,
        "best_epoch": best_epoch if best_epoch >= start_epoch else None,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "epochs_completed": len(history["train_loss"]),
        "device": config.device,
        "data_dir": config.data_dir,
        "checkpoint": config.checkpoint,
        "artifacts": {
            "history_json": history_path,
            "history_csv": history_path.with_suffix(".csv"),
            "best_checkpoint": config.checkpoint,
        },
    }
    summary_path = history_path.with_suffix(".summary.json")
    export_training_summary(summary, summary_path)
    if progress:
        progress(config.epochs, config.epochs, "message", None, None, "Hoan tat huan luyen.")
    return history
