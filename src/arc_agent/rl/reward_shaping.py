"""Configurable reward shaping for high-level subgoal PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RewardBreakdown:
    total: float
    external: float
    information_gain: float = 0.0
    causal_relation: float = 0.0
    confidence: float = 0.0
    success: float = 0.0
    subgoal: float = 0.0
    step_penalty: float = 0.0
    repeated_state: float = 0.0
    hazard: float = 0.0


class RewardShaper:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.success_bonus = float(config.get("success_bonus", 5.0))
        self.information_gain_bonus = float(config.get("information_gain_bonus", 0.2))
        self.causal_relation_bonus = float(config.get("causal_relation_bonus", 0.2))
        self.confidence_bonus = float(config.get("confidence_bonus", 0.05))
        self.subgoal_bonus = float(config.get("subgoal_bonus", 0.1))
        self.step_penalty = float(config.get("step_penalty", 0.01))
        self.repeated_state_penalty = float(config.get("repeated_state_penalty", 0.05))
        self.hazard_penalty = float(config.get("hazard_penalty", 2.0))

    def shape(
        self,
        external_reward: float,
        done: bool,
        info_gain: float = 0.0,
        new_causal_edges: int = 0,
        confidence_delta: float = 0.0,
        reached_subgoal: bool = False,
        repeated_state: bool = False,
        hazard: bool = False,
    ) -> RewardBreakdown:
        breakdown = RewardBreakdown(total=external_reward, external=external_reward)
        breakdown.information_gain = self.information_gain_bonus * info_gain
        breakdown.causal_relation = self.causal_relation_bonus * new_causal_edges
        breakdown.confidence = self.confidence_bonus * confidence_delta
        breakdown.success = self.success_bonus if done and external_reward > 0 else 0.0
        breakdown.subgoal = self.subgoal_bonus if reached_subgoal else 0.0
        breakdown.step_penalty = -self.step_penalty
        breakdown.repeated_state = -self.repeated_state_penalty if repeated_state else 0.0
        breakdown.hazard = -self.hazard_penalty if hazard else 0.0
        breakdown.total = sum(
            [
                breakdown.external,
                breakdown.information_gain,
                breakdown.causal_relation,
                breakdown.confidence,
                breakdown.success,
                breakdown.subgoal,
                breakdown.step_penalty,
                breakdown.repeated_state,
                breakdown.hazard,
            ]
        )
        return breakdown
