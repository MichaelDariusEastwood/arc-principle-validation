from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class UpdateControls:
    lr_scale: float
    momentum: float
    weight_decay: float
    grad_clip: float
    noise_scale: float


class LearnedOptimizer(nn.Module):
    """A small recurrent meta-controller that emits update-rule controls.

    This is intentionally lightweight so it can run on laptop hardware while
    still being a *real* learned update rule whose own parameters can change.
    """

    def __init__(self, feature_dim: int = 8, hidden_size: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(feature_dim, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 5),
        )
        self.reset_state(batch_size=1)

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        device = device or next(self.parameters()).device
        self.h = torch.zeros(batch_size, self.hidden_size, device=device)
        self.c = torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, features: torch.Tensor) -> UpdateControls:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        self.h, self.c = self.cell(features, (self.h, self.c))
        raw = self.head(self.h)
        lr_scale = torch.sigmoid(raw[..., 0]) * 2.0 + 0.05
        momentum = torch.sigmoid(raw[..., 1]) * 0.98
        weight_decay = torch.sigmoid(raw[..., 2]) * 5.0e-3
        grad_clip = torch.sigmoid(raw[..., 3]) * 9.5 + 0.5
        noise_scale = torch.sigmoid(raw[..., 4]) * 1.0e-2
        return UpdateControls(
            lr_scale=float(lr_scale.squeeze(0).item()),
            momentum=float(momentum.squeeze(0).item()),
            weight_decay=float(weight_decay.squeeze(0).item()),
            grad_clip=float(grad_clip.squeeze(0).item()),
            noise_scale=float(noise_scale.squeeze(0).item()),
        )

    def clone_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_cloned_state(self, state: dict) -> None:
        self.load_state_dict({k: v.detach().clone() for k, v in state.items()})
