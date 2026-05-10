"""PPO update implementation."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

from arc_agent.rl.rollout_buffer import RolloutBuffer


class PPOTrainer:
    def __init__(self, actor_critic: Any, optimizer: Any, config: dict[str, Any]) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for PPOTrainer.")
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.clip_coef = float(config.get("clip_coef", 0.2))
        self.ent_coef = float(config.get("ent_coef", 0.02))
        self.vf_coef = float(config.get("vf_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))
        self.update_epochs = int(config.get("update_epochs", 4))
        self.minibatch_size = int(config.get("minibatch_size", 512))
        self.target_kl = config.get("target_kl")

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        data = buffer.tensors()
        batch_size = data["actions"].shape[0]
        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }
        for _ in range(self.update_epochs):
            indices = torch.randperm(batch_size, device=data["actions"].device)
            for start in range(0, batch_size, self.minibatch_size):
                mb = indices[start : start + self.minibatch_size]
                log_prob, entropy, value = self.actor_critic.evaluate_actions(
                    data["graph_embeddings"][mb],
                    data["belief_features"][mb],
                    data["actions"][mb],
                )
                ratio = (log_prob - data["old_log_probs"][mb]).exp()
                advantage = data["advantages"][mb]
                unclipped = -advantage * ratio
                clipped = -advantage * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = torch.max(unclipped, clipped).mean()
                value_loss = 0.5 * (data["returns"][mb] - value).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = (data["old_log_probs"][mb] - log_prob).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                stats["policy_loss"].append(float(policy_loss.detach().cpu()))
                stats["value_loss"].append(float(value_loss.detach().cpu()))
                stats["entropy"].append(float(entropy_loss.detach().cpu()))
                stats["approx_kl"].append(float(approx_kl.detach().cpu()))
                stats["clip_fraction"].append(float(clip_fraction.detach().cpu()))
                stats["grad_norm"].append(float(grad_norm.detach().cpu()))
            if self.target_kl is not None and stats["approx_kl"] and stats["approx_kl"][-1] > float(self.target_kl):
                break
        return {key: sum(values) / len(values) if values else 0.0 for key, values in stats.items()}
