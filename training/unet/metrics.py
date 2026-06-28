"""Segmentation metrics computed on a held-out validation set.

For wire segmentation the headline number is Dice, but precision (no
hallucinated wires) and recall (no broken traces) matter more for downstream
graph recovery, so we report all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SegMetrics:
    dice: float
    precision: float
    recall: float

    def __str__(self) -> str:
        return f"dice={self.dice:.4f} precision={self.precision:.4f} recall={self.recall:.4f}"


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> SegMetrics:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return SegMetrics(dice=dice, precision=precision, recall=recall)
