"""Information-gain heuristics for candidate subgoals."""

from __future__ import annotations

from arc_agent.belief.belief_state import BeliefState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES


def object_uncertainty(belief: BeliefState, object_id: int) -> float:
    object_belief = belief.object_beliefs.get(object_id)
    if not object_belief:
        return 1.0
    uncertainties = [1.0 - abs(float(object_belief.get(name, 0.5)) - 0.5) * 2.0 for name in AFFORDANCE_NAMES]
    return sum(uncertainties) / len(uncertainties)
