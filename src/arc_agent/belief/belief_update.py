"""Deterministic/probabilistic belief update rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agent.belief.belief_state import BeliefState
from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES


@dataclass
class TransitionDelta:
    contact_object_id: int | None = None
    failed_movement_blocker: int | None = None
    disappeared_object_ids: list[int] = field(default_factory=list)
    moved_object_ids: list[int] = field(default_factory=list)
    remote_changed_object_ids: list[int] = field(default_factory=list)
    reward_delta: float = 0.0
    done: bool = False
    death_like: bool = False
    state_hash: str | None = None


class BeliefUpdater:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.initial_probability = float(config.get("initial_probability", 0.1))
        self.alpha = float(config.get("alpha", 0.35))
        self.beta = float(config.get("beta", 0.2))
        self.min_probability = float(config.get("min_probability", 0.01))
        self.max_probability = float(config.get("max_probability", 0.99))

    def update(
        self,
        belief: BeliefState,
        graph: GraphState | None = None,
        gnn_output: Any | None = None,
        delta: TransitionDelta | None = None,
        last_reward: float = 0.0,
        last_done: bool = False,
    ) -> BeliefState:
        delta = delta or TransitionDelta(reward_delta=last_reward, done=last_done)
        if graph is not None:
            self._ensure_graph_objects(belief, graph)
            self._blend_gnn_priors(belief, graph, gnn_output)
        if delta.failed_movement_blocker is not None:
            self.increase(belief, delta.failed_movement_blocker, "blocking", 1.0)
            belief.blocked_cells.add(self._object_position(graph, delta.failed_movement_blocker))
        if delta.contact_object_id is not None:
            belief.unknown_objects.discard(delta.contact_object_id)
        for object_id in delta.disappeared_object_ids:
            self.increase(belief, object_id, "collectible", 1.0)
            self.decrease(belief, object_id, "blocking", 0.3)
        for object_id in delta.moved_object_ids:
            self.increase(belief, object_id, "movable", 1.0)
        if delta.contact_object_id is not None:
            for object_id in delta.remote_changed_object_ids:
                edge = belief.causal_edges.setdefault((delta.contact_object_id, object_id), {"probability": 0.1, "evidence_count": 0.0})
                old = float(edge.get("probability", 0.1))
                edge["probability"] = self._clamp(old + self.alpha * (1.0 - old))
                edge["evidence_count"] = float(edge.get("evidence_count", 0.0)) + 1.0
                self.increase(belief, delta.contact_object_id, "trigger", 0.7)
        if delta.contact_object_id is not None and (delta.reward_delta > 0 or delta.done):
            self.increase(belief, delta.contact_object_id, "goal_related", 1.0)
            belief.goals.add(delta.contact_object_id)
        if delta.death_like and delta.contact_object_id is not None:
            self.increase(belief, delta.contact_object_id, "hazardous", 1.0)
            belief.hazards.add(delta.contact_object_id)
        if delta.state_hash:
            belief.visited_states.add(delta.state_hash)
        if graph is not None and graph.action_context is not None and float(graph.action_context.sum()) > 0:
            belief.last_action = int(graph.action_context.argmax().item())
        belief.step_count += 1
        return belief

    def increase(self, belief: BeliefState, object_id: int, name: str, evidence_strength: float) -> float:
        object_belief = belief.ensure_object(object_id, self.initial_probability)
        old = object_belief.get(name, self.initial_probability)
        new = self._clamp(old + self.alpha * evidence_strength * (1.0 - old))
        object_belief[name] = new
        return new

    def decrease(self, belief: BeliefState, object_id: int, name: str, evidence_strength: float) -> float:
        object_belief = belief.ensure_object(object_id, self.initial_probability)
        old = object_belief.get(name, self.initial_probability)
        new = self._clamp(old * (1.0 - self.beta * evidence_strength))
        object_belief[name] = new
        return new

    def _ensure_graph_objects(self, belief: BeliefState, graph: GraphState) -> None:
        for node in graph.nodes:
            belief.ensure_object(node.track_id or node.id, self.initial_probability)

    def _blend_gnn_priors(self, belief: BeliefState, graph: GraphState, gnn_output: Any | None) -> None:
        probs = getattr(gnn_output, "affordance_probs", None)
        if probs is None:
            return
        try:
            values = probs.detach().cpu().numpy()
        except Exception:
            values = probs
        for node_idx, node in enumerate(graph.nodes):
            object_belief = belief.ensure_object(node.track_id or node.id, self.initial_probability)
            for name_idx, name in enumerate(AFFORDANCE_NAMES):
                if name_idx < values.shape[1]:
                    object_belief[name] = self._clamp(0.8 * object_belief.get(name, 0.1) + 0.2 * float(values[node_idx, name_idx]))

    def _object_position(self, graph: GraphState | None, object_id: int) -> tuple[int, int]:
        if graph is None:
            return (-1, -1)
        for node in graph.nodes:
            if (node.track_id or node.id) == object_id:
                return (round(node.centroid[0]), round(node.centroid[1]))
        return (-1, -1)

    def _clamp(self, value: float) -> float:
        return min(self.max_probability, max(self.min_probability, value))


def compare_graphs(
    previous_graph: GraphState | None,
    current_graph: GraphState,
    last_action: int | None = None,
    reward_delta: float = 0.0,
    done: bool = False,
) -> TransitionDelta:
    if previous_graph is None:
        return TransitionDelta(reward_delta=reward_delta, done=done)
    previous_ids = {node.track_id or node.id: node for node in previous_graph.nodes}
    current_ids = {node.track_id or node.id: node for node in current_graph.nodes}
    disappeared = [object_id for object_id in previous_ids if object_id not in current_ids]
    moved: list[int] = []
    for object_id, node in current_ids.items():
        prev = previous_ids.get(object_id)
        if prev is not None and node.centroid != prev.centroid:
            moved.append(object_id)
    contact = moved[0] if moved else (disappeared[0] if disappeared else None)
    remote = [object_id for object_id in moved if object_id != contact]
    return TransitionDelta(
        contact_object_id=contact,
        disappeared_object_ids=disappeared,
        moved_object_ids=moved,
        remote_changed_object_ids=remote,
        reward_delta=reward_delta,
        done=done,
    )
