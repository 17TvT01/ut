from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from torch import amp


def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()


def train_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[amp.GradScaler] = None,
    non_blocking: bool = False,
) -> float:
    model.train()
    if scaler is None:
        scaler = amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    for batch in dataloader:
        volume = batch["volume"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        mask = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(volume)
            loss = dice_loss(logits, mask)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
    return total_loss / max(len(dataloader), 1)


def evaluate_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            volume = batch["volume"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            mask = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(volume)
                loss = dice_loss(logits, mask)
            total_loss += float(loss.detach().item())
    return total_loss / max(len(dataloader), 1)


__all__ = ["dice_loss", "train_epoch", "evaluate_epoch"]
