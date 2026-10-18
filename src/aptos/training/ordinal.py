"""Research-only cumulative-link ordinal severity head.

Four strictly ordered, trainable cutpoints share one latent severity score.
P(Y > k | image) = sigmoid(latent - cutpoint[k]); k=0,1,2,3.
This cannot be loaded as the legacy scalar-regression serving checkpoint.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from aptos.config import N_GRADES


class CumulativeOrdinalHead(nn.Module):
    """Rank-consistent cumulative logits with a shared one-output backbone."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.base_cut = nn.Parameter(torch.tensor(-1.5))
        # softplus(0.5413) ~= 1.0, yielding initial -1.5,-0.5,0.5,1.5.
        self.cut_gaps = nn.Parameter(torch.full((N_GRADES - 2,), 0.54132485))

    def cutpoints(self) -> torch.Tensor:
        gaps = F.softplus(self.cut_gaps) + 1e-4
        return torch.cat((self.base_cut.view(1),
                          self.base_cut + gaps.cumsum(dim=0)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(images)
        if latent.ndim != 2 or latent.shape[1] != 1:
            raise ValueError("ordinal backbone must return shape [batch, 1]")
        return latent - self.cutpoints().unsqueeze(0)


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    """Encode 0..4 severity as four binary indicators: grade > 0,1,2,3."""
    if labels.ndim != 1 or labels.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError("ordinal labels must be a 1-D integer tensor")
    if labels.numel() == 0 or ((labels < 0) | (labels >= N_GRADES)).any():
        raise ValueError("ordinal labels must be 0..4")
    return (labels[:, None] > torch.arange(N_GRADES - 1, device=labels.device)).float()


def ordinal_expected_grade(logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative logits to continuous expected grade in [0,4]."""
    if logits.ndim != 2 or logits.shape[1] != N_GRADES - 1:
        raise ValueError("ordinal logits must have shape [batch, 4]")
    return torch.sigmoid(logits.float()).sum(dim=1)
