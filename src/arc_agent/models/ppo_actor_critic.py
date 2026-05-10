"""PPO actor-critic over high-level candidate strategies."""

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
class ActorCriticOutput:
    logits: Any
    value: Any


class PPOActorCritic(nn.Module if nn is not None else object):
    def __init__(
        self,
        graph_embedding_dim: int,
        belief_feature_dim: int,
        hidden_dim: int = 256,
        num_actions: int = 8,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for PPOActorCritic.")
        super().__init__()
        input_dim = graph_embedding_dim + belief_feature_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, graph_embedding: Any, belief_features: Any, action_mask: Any | None = None) -> ActorCriticOutput:
        if torch is None:
            raise RuntimeError("PyTorch is required for PPOActorCritic.")
        if graph_embedding.ndim == 1:
            graph_embedding = graph_embedding.unsqueeze(0)
        if belief_features.ndim == 1:
            belief_features = belief_features.unsqueeze(0)
        hidden = self.backbone(torch.cat([graph_embedding, belief_features], dim=-1))
        logits = self.policy(hidden)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        value = self.value(hidden).squeeze(-1)
        return ActorCriticOutput(logits=logits, value=value)

    def act(self, graph_embedding: Any, belief_features: Any, action_mask: Any | None = None) -> tuple[Any, Any, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for PPOActorCritic.")
        output = self.forward(graph_embedding, belief_features, action_mask)
        dist = torch.distributions.Categorical(logits=output.logits)
        action = dist.sample()
        return action, dist.log_prob(action), output.value

    def evaluate_actions(self, graph_embedding: Any, belief_features: Any, actions: Any, action_mask: Any | None = None) -> tuple[Any, Any, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for PPOActorCritic.")
        output = self.forward(graph_embedding, belief_features, action_mask)
        dist = torch.distributions.Categorical(logits=output.logits)
        return dist.log_prob(actions), dist.entropy(), output.value
