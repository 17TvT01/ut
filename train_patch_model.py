#!/usr/bin/env python
"""
Train Simplified UNet on Patch Dataset
Strategy: Small model + many patches + heavy augmentation
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import amp

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.patch_dataset import create_patch_datasets
from nodule_ai.simple_model import SimpleUNet3D, count_parameters
from nodule_ai.training import dice_loss, focal_loss


def train_patch_model(
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints"),
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    loss_type: str = "dice",  # "dice" or "focal"
    patch_size: tuple = (64, 64, 64),
    num_positive: int = 15,
    num_negative: int = 45,
):
    """
    Train simplified UNet on patches.
    
    Args:
        loss_type: "dice" or "focal"
        patch_size: Size of patches
        num_positive: Number of positive patches per volume
        num_negative: Number of negative patches per volume
    """
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"simple_unet3d_patches_{loss_type}_{timestamp}"
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    
    print("=" * 80)
    print(f"TRAINING SIMPLE UNET ON PATCHES ({loss_type.upper()} LOSS)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Patch size: {patch_size}")
    print(f"Positive patches per volume: {num_positive}")
    print(f"Negative patches per volume: {num_negative}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Create datasets
    train_dataset, val_dataset = create_patch_datasets(
        data_dir,
        patch_size=patch_size,
        num_positive_per_volume=num_positive,
        num_negative_per_volume=num_negative,
        val_split=0.2,
        seed=42,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Train patches: {len(train_dataset)}")
    print(f"Val patches: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Create model
    model = SimpleUNet3D(n_channels=1, n_classes=1, base_filters=16, dropout=0.1)
    model = model.to(device)
    
    print("=" * 80)
    print("MODEL")
    print("=" * 80)
    print(f"Architecture: SimpleUNet3D")
    print(f"Parameters: {count_parameters(model):,}")
    print()
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Select loss function
    if loss_type.lower() == "focal":
        criterion = focal_loss
        print("Loss function: Focal Loss (α=0.25, γ=2.0)")
    else:
        criterion = dice_loss
        print("Loss function: Dice Loss")
    
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
    print()
    
    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 20
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            volume = batch["volume"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.float32)
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(volume)
                loss = criterion(logits, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.detach().item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                volume = batch["volume"].to(device, dtype=torch.float32)
                mask = batch["mask"].to(device, dtype=torch.float32)
                
                with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(volume)
                    loss = criterion(logits, mask)
                
                val_loss += loss.detach().item()
        
        val_loss /= len(val_loader)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping & model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            status = "✓ (best)"
        else:
            patience_counter += 1
            status = f"(patience {patience_counter}/{patience})"
        
        # Print progress
        if (epoch % 5 == 0) or (epoch == 1) or (patience_counter == 0):
            print(f"Epoch {epoch:3d}/{epochs}: "
                  f"train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f} {status}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break
    
    print()
    print("=" * 80)
    print("✓ TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved: {checkpoint_path}")
    print()
    
    # Save history
    import json
    history_path = checkpoint_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {history_path}")
    
    return model, history, checkpoint_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train simplified UNet on patches")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="dice", choices=["dice", "focal"], help="Loss function")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    model, history, checkpoint_path = train_patch_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        loss_type=args.loss,
        patch_size=(64, 64, 64),
        num_positive=15,
        num_negative=45,
    )
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Evaluate model on test set")
    print("2. If performance is good, try increasing patch extraction")
    print("3. If performance is poor, try:")
    print("   - Focal loss (if used Dice)")
    print("   - More patches per volume")
    print("   - Different patch size")
    print("   - More aggressive augmentation")
