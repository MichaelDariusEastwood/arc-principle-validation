from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


class MLPOptimizee(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int, init_std: float = 0.05):
        super().__init__()
        dims = [input_dim, *hidden_sizes, output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float = 0.05) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def clone_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_cloned_state(self, state: dict) -> None:
        self.load_state_dict({k: v.detach().clone() for k, v in state.items()})

    def parameter_norm(self) -> float:
        total = 0.0
        for p in self.parameters():
            total += float(torch.norm(p.detach()).item())
        return total
