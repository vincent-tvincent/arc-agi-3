"""Candidate high-level subgoal generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from arc_agent.belief.belief_state import BeliefState
from arc_agent.graph.graph_data import GraphState
from arc_agent.planning.information_gain import object_uncertainty


class CandidateType(IntEnum):
    REACH_KNOWN_GOAL = 0
    TEST_UNKNOWN_OBJECT = 1
    TEST_BLOCKING_OBJECT = 2
    COLLECT_COLLECTIBLE_LIKE_OBJECT = 3
    TEST_PUSHABLE_OBJECT = 4
    ACTIVATE_TRIGGER_LIKE_OBJECT = 5
    RETRY_PREVIOUSLY_BLOCKED_PATH = 6
    EXPLORE_FRONTIER = 7
    AVOID_HAZARD_ROUTE = 8


@dataclass
class CandidateSubgoal:
    name: str
    target_object_id: int | None
    target_position: tuple[int, int] | None
    expected_information_gain: float
    expected_goal_progress: float
    expected_risk: float
    planner_path: list[int]
    features: np.ndarray
    symbolic_score: float = 0.0
    ppo_score: float = 0.0


class CandidateGenerator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.max_candidates = int(config.get("max_candidates", 32))

    def generate(self, belief: BeliefState, graph: GraphState, gnn_output: Any | None = None) -> list[CandidateSubgoal]:
        candidates: list[CandidateSubgoal] = []
        for node in graph.nodes:
            object_id = node.track_id or node.id
            object_belief = belief.object_beliefs.get(object_id, {})
            position = (round(node.centroid[0]), round(node.centroid[1]))
            risk = float(object_belief.get("hazardous", 0.0))
            uncertainty = object_uncertainty(belief, object_id)
            if object_id in belief.goals or object_belief.get("goal_related", 0.0) > 0.55:
                candidates.append(self._candidate(CandidateType.REACH_KNOWN_GOAL, object_id, position, uncertainty, 1.0, risk))
            if object_id in belief.unknown_objects or uncertainty > 0.5:
                candidates.append(self._candidate(CandidateType.TEST_UNKNOWN_OBJECT, object_id, position, uncertainty, 0.2, risk))
            if object_belief.get("blocking", 0.0) > 0.45:
                candidates.append(self._candidate(CandidateType.TEST_BLOCKING_OBJECT, object_id, position, uncertainty, 0.1, risk))
            if object_belief.get("collectible", 0.0) > 0.4:
                candidates.append(self._candidate(CandidateType.COLLECT_COLLECTIBLE_LIKE_OBJECT, object_id, position, uncertainty, 0.6, risk))
            if object_belief.get("movable", 0.0) > 0.4:
                candidates.append(self._candidate(CandidateType.TEST_PUSHABLE_OBJECT, object_id, position, uncertainty, 0.3, risk))
            if object_belief.get("trigger", 0.0) > 0.35:
                candidates.append(self._candidate(CandidateType.ACTIVATE_TRIGGER_LIKE_OBJECT, object_id, position, uncertainty, 0.5, risk))

        if not candidates and graph.nodes:
            node = graph.nodes[0]
            candidates.append(self._candidate(CandidateType.EXPLORE_FRONTIER, node.track_id or node.id, (round(node.centroid[0]), round(node.centroid[1])), 1.0, 0.1, 0.0))
        return sorted(candidates, key=lambda item: (item.expected_goal_progress, item.expected_information_gain), reverse=True)[
            : self.max_candidates
        ]

    def _candidate(
        self,
        kind: CandidateType,
        target_object_id: int | None,
        target_position: tuple[int, int] | None,
        info_gain: float,
        goal_progress: float,
        risk: float,
    ) -> CandidateSubgoal:
        features = np.zeros((12,), dtype=np.float32)
        features[0] = float(kind)
        features[1] = float(target_object_id or -1)
        features[2] = float(target_position[0] if target_position else -1)
        features[3] = float(target_position[1] if target_position else -1)
        features[4] = float(info_gain)
        features[5] = float(goal_progress)
        features[6] = float(risk)
        features[7] = 1.0 if target_object_id is not None else 0.0
        return CandidateSubgoal(
            name=kind.name,
            target_object_id=target_object_id,
            target_position=target_position,
            expected_information_gain=info_gain,
            expected_goal_progress=goal_progress,
            expected_risk=risk,
            planner_path=[],
            features=features,
        )
