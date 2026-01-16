#!/usr/bin/env python
"""
Phase 2: Retrain với Focal Loss
- Thay Dice Loss → Focal Loss
- Tăng epochs: 20 → 100
- Thêm learning rate scheduler
- Thêm early stopping
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch import amp

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D
from nodule_ai.training import focal_loss, train_epoch, evaluate_epoch
from nodule_ai.trainer import TrainingConfig, set_seed, export_training_history


def train_with_focal_loss(
    config: TrainingConfig,
    loss_fn_name: str = "focal",
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
):
    """
    Train model với Focal Loss hoặc loss function khác.
    
    Args:
        config: Training configuration
        loss_fn_name: "focal", "weighted_dice", hoặc "tversky"
        learning_rate: Override learning rate
        epochs: Override epochs
    """
    
    set_seed(config.seed)
    device = torch.device(config.device)
    
    print("=" * 80)
    print(f"PHASE 2: RETRAIN WITH {loss_fn_name.upper()} LOSS")
    print("=" * 80)
    print()
    
    # Create output directory
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    
    # Build model
    print("Building model...")
    model = ComplexUNet3D(
        n_channels=config.n_channels,
        n_classes=config.n_classes,
        base_filters=config.base_filters,
        dropout=config.dropout,
        upsample_mode=config.upsample_mode,
    )
    model = model.to(device)
    print(f"✓ Model: {config.checkpoint.stem}")
    print()
    
    # Build dataloaders
    print("Building dataloaders...")
    dataset = LIDCDataset(config.data_dir, cache=False, target_shape=config.target_shape)
    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    print(f"✓ Train: {train_size}, Val: {val_size}")
    print()
    
    # Setup training
    lr = learning_rate or config.learning_rate
    epochs_to_train = epochs or config.epochs
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    scaler = amp.GradScaler(enabled=config.use_amp)
    
    # Select loss function
    if loss_fn_name.lower() == "focal":
        loss_fn = focal_loss
        print(f"Loss: Focal Loss (α=0.25, γ=2.0)")
    elif loss_fn_name.lower() == "weighted_dice":
        from nodule_ai.training import weighted_dice_loss
        loss_fn = weighted_dice_loss
        print(f"Loss: Weighted Dice Loss (pos_weight=2.0)")
    elif loss_fn_name.lower() == "tversky":
        from nodule_ai.training import tversky_loss
        loss_fn = tversky_loss
        print(f"Loss: Tversky Loss (α=0.5, β=0.5)")
    else:
        from nodule_ai.training import dice_loss
        loss_fn = dice_loss
        print(f"Loss: Dice Loss")
    
    print(f"Optimizer: Adam (lr={lr})")
    print(f"Epochs: {epochs_to_train}")
    print(f"Batch size: {config.batch_size}")
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
    
    for epoch in range(1, epochs_to_train + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            volume = batch["volume"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(device_type=device.type, enabled=config.use_amp or False):
                logits = model(volume)
                if loss_fn_name.lower() == "weighted_dice":
                    loss = loss_fn(logits, mask, pos_weight=2.0)
                else:
                    loss = loss_fn(logits, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += float(loss.detach().item())
        
        train_loss /= max(len(train_loader), 1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                volume = batch["volume"].to(device=device, dtype=torch.float32)
                mask = batch["mask"].to(device=device, dtype=torch.float32)
                
                with amp.autocast(device_type=device.type, enabled=config.use_amp or False):
                    logits = model(volume)
                    if loss_fn_name.lower() == "weighted_dice":
                        loss = loss_fn(logits, mask, pos_weight=2.0)
                    else:
                        loss = loss_fn(logits, mask)
                
                val_loss += float(loss.detach().item())
        
        val_loss /= max(len(val_loader), 1)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.checkpoint)
            status = "✓ (best)"
        else:
            patience_counter += 1
            status = f"(patience {patience_counter}/{patience})"
        
        if (epoch % 10 == 0) or (epoch == 1) or (patience_counter == 0):
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} {status}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break
    
    print()
    print("=" * 80)
    print("✓ TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved: {config.checkpoint}")
    print()
    
    # Save history
    history_json = config.checkpoint.with_suffix(".focal.history.json")
    export_training_history(history, history_json)
    print(f"History saved: {history_json}")
    
    return model, history


if __name__ == "__main__":
    config = TrainingConfig(
        data_dir=Path("data"),
        epochs=100,
        batch_size=1,
        learning_rate=1e-3,
        val_split=0.2,
        seed=42,
        device="cuda",
        checkpoint=Path("checkpoints/complex_unet3d_focal.pt"),
    )
    
    train_with_focal_loss(config, loss_fn_name="focal", epochs=100)
