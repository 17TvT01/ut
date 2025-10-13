from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nodule_ai.trainer import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lung nodule model")
    parser.add_argument("data_dir", type=Path, help="Root directory containing LIDC-style studies")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (3D data is memory heavy)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/complex_unet3d.pt"),
        help="Path to save best checkpoint",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cuda or cpu)",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="use_amp", action="store_true", help="Force enable AMP mixed precision")
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable AMP even on GPU")
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable pinned memory for faster host-to-device transfers",
    )
    pin_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory in the DataLoader",
    )
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    parser.add_argument(
        "--base-filters",
        type=int,
        default=32,
        help="Number of base convolution filters (applies to all supported models)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate applied to ComplexUNet3D residual blocks",
    )
    parser.add_argument(
        "--upsample-mode",
        type=str,
        default="trilinear",
        choices=["trilinear", "nearest"],
        help="Upsample mode for complex_unet3d decoder blocks",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=1,
        help="Number of input channels for ComplexUNet3D",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=1,
        help="Number of output classes for ComplexUNet3D",
    )
    parser.set_defaults(use_amp=None, pin_memory=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        base_filters=args.base_filters,
        dropout=args.dropout,
        upsample_mode=args.upsample_mode,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        resume=args.resume,
        use_amp=args.use_amp,
        pin_memory=args.pin_memory,
    )
    history = train_model(config)
    print("Training complete. Loss history:")
    for i, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
        print(f"Epoch {i}: train={train_loss:.4f}, val={val_loss:.4f}")
    print(f"Best checkpoint saved at {config.checkpoint}")
    history_path = config.checkpoint.with_suffix(".history.json")
    summary_path = history_path.with_suffix(".summary.json")
    print(f"History exported to {history_path}")
    print(f"Metrics summary exported to {summary_path}")


if __name__ == "__main__":
    main()
