"""Shared torch encoders."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


class MLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2, dropout: float = 0.0) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for MLP.")
        super().__init__()
        modules: list[Any] = []
        dim = input_dim
        for _ in range(max(layers - 1, 1)):
            modules.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        modules.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Any) -> Any:
        return self.net(x)
