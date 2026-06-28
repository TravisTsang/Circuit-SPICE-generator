"""Loss functions for thin-line binary segmentation.

BCE keeps per-pixel calibration; Dice handles the extreme foreground/background
imbalance of thin wires. The combination is a strong default. Focal loss is
available for cases where the model misses faint traces.
"""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: float | None = None) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)
