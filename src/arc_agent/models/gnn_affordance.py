"""Custom message-passing GNN for affordance prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


@dataclass
class GNNOutput:
    node_embeddings: Any
    graph_embedding: Any
    affordance_logits: Any
    affordance_probs: Any


class MessagePassingLayer(nn.Module if nn is not None else object):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for MessagePassingLayer.")
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is required for MessagePassingLayer.")
        if edge_index.numel() == 0:
            return self.norm(x)
        source, target = edge_index[0], edge_index[1]
        messages = self.message(torch.cat([x[source], x[target], edge_attr], dim=-1))
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, target, messages)
        degrees = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        degrees.index_add_(0, target, torch.ones((messages.shape[0], 1), dtype=x.dtype, device=x.device))
        aggregate = aggregate / degrees.clamp_min(1.0)
        return self.norm(self.update(torch.cat([x, aggregate], dim=-1)) + x)


class GNNAffordanceModel(nn.Module if nn is not None else object):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        num_affordances: int = 9,
        dropout: float = 0.1,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for GNNAffordanceModel.")
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim, edge_in_dim, dropout) for _ in range(layers)])
        self.affordance_head = nn.Linear(hidden_dim, num_affordances)
        self.pool_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, graph: Any) -> GNNOutput:
        if torch is None:
            raise RuntimeError("PyTorch is required for GNNAffordanceModel.")
        x = self.node_encoder(graph.node_features)
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.edge_attr)
        logits = self.affordance_head(x)
        probs = torch.sigmoid(logits)
        if x.shape[0] == 0:
            graph_embedding = torch.zeros((self.affordance_head.in_features,), dtype=x.dtype, device=x.device)
        else:
            weights = self.pool_gate(x)
            graph_embedding = (x * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)
        return GNNOutput(node_embeddings=x, graph_embedding=graph_embedding, affordance_logits=logits, affordance_probs=probs)
