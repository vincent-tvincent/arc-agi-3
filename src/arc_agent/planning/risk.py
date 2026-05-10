"""Risk map helpers."""

from __future__ import annotations

from arc_agent.belief.belief_state import BeliefState
from arc_agent.graph.graph_data import GraphState


def build_risk_map(belief: BeliefState, graph: GraphState, threshold: float = 0.2) -> dict[tuple[int, int], float]:
    risk: dict[tuple[int, int], float] = {}
    for node in graph.nodes:
        object_id = node.track_id or node.id
        p = belief.object_beliefs.get(object_id, {}).get("hazardous", 0.0)
        if p >= threshold:
            for x, y in node.cells:
                risk[(x, y)] = max(risk.get((x, y), 0.0), float(p))
    return risk


def blocked_cells_from_belief(belief: BeliefState, graph: GraphState, threshold: float = 0.65) -> set[tuple[int, int]]:
    blocked = set(belief.blocked_cells)
    for node in graph.nodes:
        object_id = node.track_id or node.id
        p = belief.object_beliefs.get(object_id, {}).get("blocking", 0.0)
        if p >= threshold:
            blocked.update(node.cells)
    return blocked
