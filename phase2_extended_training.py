#!/usr/bin/env python
"""
Extended Training Phase 2 - Với cấu hình tốt hơn
- Tăng epochs
- Fine-tuning hyperparameters
- Better initialization
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch import amp

sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D
from nodule_ai.training import focal_loss
from nodule_ai.trainer import TrainingConfig, set_seed, export_training_history


def train_extended(epochs: int = 200, learning_rate: float = 5e-4):
    """
    Extended training với Focal Loss.
    Giảm LR để training sâu hơn.
    """
    
    config = TrainingConfig(
        data_dir=Path("data"),
        epochs=epochs,
        batch_size=1,
        learning_rate=learning_rate,
        val_split=0.2,
        seed=42,
        device="cuda",
        checkpoint=Path("checkpoints/complex_unet3d_focal_extended.pt"),
    )
    
    set_seed(config.seed)
    device = torch.device(config.device)
    
    print("=" * 80)
    print(f"EXTENDED TRAINING: FOCAL LOSS (Epochs={epochs}, LR={learning_rate})")
    print("=" * 80)
    print()
    
    # Build model
    model = ComplexUNet3D(
        n_channels=1, n_classes=1, base_filters=16,
        dropout=0.1, upsample_mode="trilinear"
    )
    model = model.to(device)
    
    # Build dataloaders
    dataset = LIDCDataset(config.data_dir, cache=False, target_shape=config.target_shape)
    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    print()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = amp.GradScaler(enabled=True)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 30
    
    print(f"Loss: Focal Loss")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Early stopping: patience={patience}")
    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            volume = batch["volume"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.float32)
            
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=True):
                logits = model(volume)
                loss = focal_loss(logits, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
        
        train_loss /= max(len(train_loader), 1)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                volume = batch["volume"].to(device, dtype=torch.float32)
                mask = batch["mask"].to(device, dtype=torch.float32)
                with amp.autocast(device_type="cuda", enabled=True):
                    logits = model(volume)
                    loss = focal_loss(logits, mask)
                val_loss += float(loss.item())
        
        val_loss /= max(len(val_loader), 1)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.checkpoint)
            status = "✓ (best)"
        else:
            patience_counter += 1
            status = f"(patience {patience_counter}/{patience})"
        
        if epoch % 20 == 0 or epoch == 1 or patience_counter == 0:
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f} {status}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break
    
    print()
    print("=" * 80)
    print(f"✓ Training completed: {config.checkpoint}")
    print()
    
    # Export history
    history_file = config.checkpoint.with_suffix(".history.json")
    export_training_history(history, history_file)
    
    return config.checkpoint


if __name__ == "__main__":
    # Train với cấu hình tốt hơn
    checkpoint = train_extended(epochs=200, learning_rate=5e-4)
    
    print(f"\n✓ Model saved: {checkpoint}")
    print("\nNext: python evaluate_phase2.py")
