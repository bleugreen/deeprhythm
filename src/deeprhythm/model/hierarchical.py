"""Hierarchical tempo model separating rhythmic rate from metrical level."""

import torch
from torch import nn

from deeprhythm.model.frame_cnn import DeepRhythmModel

BASE_MIN = 60.0
BASE_WIDTH = 0.5
BASE_CLASSES = 120
LEVEL_FACTORS = (0.5, 1.0, 2.0, 4.0)


class ResidualLevelHead(nn.Module):
    """Nonlinear metrical correction initialized to preserve existing logits."""

    def __init__(self, embedding_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(LEVEL_FACTORS)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, embedding):
        return self.network(embedding)


def tempo_to_hierarchy(tempos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode BPM as a half-BPM base rate in [60, 120) and an octave level."""
    tempos = torch.as_tensor(tempos, dtype=torch.float32)
    levels = torch.zeros_like(tempos, dtype=torch.long)
    bases = tempos.clone()
    below = bases < BASE_MIN
    bases[below] *= 2.0
    levels[below] = 0
    normal = (~below) & (bases < 120.0)
    levels[normal] = 1
    double = (bases >= 120.0) & (bases < 240.0)
    bases[double] /= 2.0
    levels[double] = 2
    quadruple = bases >= 240.0
    bases[quadruple] /= 4.0
    levels[quadruple] = 3
    base_classes = ((bases - BASE_MIN) / BASE_WIDTH).round().long().clamp(0, BASE_CLASSES - 1)
    return base_classes, levels


def hierarchy_to_tempo(base_classes: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Compose base-rate and metrical-level classes into BPM."""
    bases = BASE_MIN + torch.as_tensor(base_classes, dtype=torch.float32) * BASE_WIDTH
    factors = torch.tensor(LEVEL_FACTORS, device=bases.device)
    return bases * factors[torch.as_tensor(levels, dtype=torch.long, device=bases.device)]


class HierarchicalDeepRhythmModel(nn.Module):
    """One HCQM backbone with independent base-rate and metrical-level heads."""

    def __init__(self, backbone=None, *, residual_level=False):
        super().__init__()
        self.backbone = backbone or DeepRhythmModel()
        self.base_head = nn.Linear(256, BASE_CLASSES)
        self.level_head = nn.Linear(256, len(LEVEL_FACTORS))
        self.level_residual = ResidualLevelHead() if residual_level else None

    def forward(self, inputs):
        embedding = self.backbone.encode(inputs)
        level_logits = self.level_head(embedding)
        if self.level_residual is not None:
            level_logits = level_logits + self.level_residual(embedding)
        return self.base_head(embedding), level_logits

    def load_backbone(self, state_dict):
        """Initialize the shared representation from a conventional DeepRhythm checkpoint."""
        self.backbone.load_state_dict(state_dict)