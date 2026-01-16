from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import amp


def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Focal Loss (binary) on logits for numerical stability.

    Using BCEWithLogits to avoid sigmoid underflow; compute p_t from CE:
    FL = alpha_t * (1 - p_t)^gamma * CE, where p_t = exp(-CE)

    Args:
        pred: Logits từ model
        target: Ground truth (0 hoặc 1)
        alpha: Weight cho positive samples
        gamma: Focusing parameter (0 = cross-entropy, 2 = recommended)
        epsilon: Small value để tránh log(0)
    """
    # CE on logits (stable), per-element
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    # p_t = prob of true class; exp(-CE) since CE = -log(p_t)
    p_t = torch.exp(-ce)
    # alpha weighting per-pixel, stays on correct device/dtype via broadcasting
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    # focal modulation
    focal = alpha_t * (1.0 - p_t).clamp(min=0.0, max=1.0) ** gamma * ce
    return focal.mean()


def weighted_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted Dice Loss: Cân bằng positive/negative samples
    
    Args:
        pred: Logits từ model
        target: Ground truth
        pos_weight: Weight cho positive samples (> 1 để tăng weight cho nốts)
        epsilon: Small value
    """
    pred = torch.sigmoid(pred)
    
    # Tính weighted intersection và union
    weights = torch.where(target == 1, pos_weight, 1.0)
    
    weighted_intersection = (pred * target * weights).sum(dim=(1, 2, 3, 4))
    weighted_pred_sum = (pred * weights).sum(dim=(1, 2, 3, 4))
    weighted_target_sum = (target * weights).sum(dim=(1, 2, 3, 4))
    
    dice = (2 * weighted_intersection + epsilon) / (
        weighted_pred_sum + weighted_target_sum + epsilon
    )
    
    return 1 - dice.mean()


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Tversky Loss: Kiểm soát tradeoff FP vs FN
    Tversky = TP / (TP + α*FP + β*FN + ε)
    
    Default: α=0.5, β=0.5 (bằng Dice)
    Tăng α: Giảm weight FP
    Tăng β: Giảm weight FN
    
    Args:
        pred: Logits từ model
        target: Ground truth
        alpha: Weight cho false positives
        beta: Weight cho false negatives
    """
    pred = torch.sigmoid(pred)
    
    tp = (pred * target).sum(dim=(1, 2, 3, 4))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3, 4))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3, 4))
    
    tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    
    return 1 - tversky.mean()


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


__all__ = ["dice_loss", "focal_loss", "weighted_dice_loss", "tversky_loss", "train_epoch", "evaluate_epoch"]
