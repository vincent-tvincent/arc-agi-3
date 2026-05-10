"""Rollout storage and GAE computation for PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except Exception:
    torch = None


@dataclass
class RolloutBuffer:
    graph_embeddings: list[Any] = field(default_factory=list)
    belief_features: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    returns: Any | None = None
    advantages: Any | None = None

    def add(
        self,
        graph_embedding: Any,
        belief_feature: Any,
        action: int,
        log_prob: Any,
        reward: float,
        done: bool,
        value: Any,
    ) -> None:
        self.graph_embeddings.append(graph_embedding.detach())
        self.belief_features.append(belief_feature.detach())
        self.actions.append(int(action))
        self.log_probs.append(log_prob.detach())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(value.detach())

    def clear(self) -> None:
        self.graph_embeddings.clear()
        self.belief_features.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.returns = None
        self.advantages = None

    def compute_returns_and_advantages(self, last_value: Any, gamma: float, gae_lambda: float) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for RolloutBuffer.")
        values = torch.stack([value.reshape(()) for value in self.values] + [last_value.reshape(())])
        advantages = torch.zeros(len(self.rewards), dtype=values.dtype, device=values.device)
        gae = torch.zeros((), dtype=values.dtype, device=values.device)
        for step in reversed(range(len(self.rewards))):
            nonterminal = 0.0 if self.dones[step] else 1.0
            delta = self.rewards[step] + gamma * values[step + 1] * nonterminal - values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
        self.advantages = advantages
        self.returns = advantages + values[:-1]

    def tensors(self) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for RolloutBuffer.")
        advantages = self.advantages
        returns = self.returns
        if advantages is None or returns is None:
            raise RuntimeError("Call compute_returns_and_advantages before tensors().")
        return {
            "graph_embeddings": torch.stack(self.graph_embeddings),
            "belief_features": torch.stack(self.belief_features),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=advantages.device),
            "old_log_probs": torch.stack([item.reshape(()) for item in self.log_probs]),
            "returns": returns,
            "advantages": (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8)),
        }
