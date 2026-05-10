"""Online belief state for object affordances and causal relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arc_agent.graph.graph_features import AFFORDANCE_NAMES


@dataclass
class BeliefState:
    object_beliefs: dict[int, dict[str, float]] = field(default_factory=dict)
    causal_edges: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    reachable_cells: set[tuple[int, int]] = field(default_factory=set)
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)
    hazards: set[int] = field(default_factory=set)
    goals: set[int] = field(default_factory=set)
    unknown_objects: set[int] = field(default_factory=set)
    visited_states: set[str] = field(default_factory=set)
    last_action: int | None = None
    step_count: int = 0

    def ensure_object(self, object_id: int, initial: float = 0.1) -> dict[str, float]:
        if object_id not in self.object_beliefs:
            self.object_beliefs[object_id] = {name: initial for name in AFFORDANCE_NAMES}
            self.unknown_objects.add(object_id)
        return self.object_beliefs[object_id]

    def mark_reachable(self, cells: set[tuple[int, int]]) -> None:
        self.reachable_cells.update(cells)

    def summary_vector(self, length: int = 32) -> np.ndarray:
        values: list[float] = [
            float(len(self.object_beliefs)),
            float(len(self.causal_edges)),
            float(len(self.reachable_cells)),
            float(len(self.blocked_cells)),
            float(len(self.hazards)),
            float(len(self.goals)),
            float(len(self.unknown_objects)),
            float(self.step_count),
        ]
        for name in AFFORDANCE_NAMES:
            probs = [belief.get(name, 0.1) for belief in self.object_beliefs.values()]
            values.append(float(sum(probs) / len(probs)) if probs else 0.0)
        if self.last_action is not None:
            values.append(float(self.last_action))
        while len(values) < length:
            values.append(0.0)
        return np.asarray(values[:length], dtype=np.float32)
