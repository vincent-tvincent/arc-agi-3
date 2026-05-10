"""Planner that combines symbolic scores with optional PPO scores."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from arc_agent.belief.belief_state import BeliefState
from arc_agent.execution.action_space import GridAction
from arc_agent.graph.graph_data import GraphState
from arc_agent.perception.object_types import ObjectComponent
from arc_agent.planning.astar import astar_path
from arc_agent.planning.candidates import CandidateSubgoal
from arc_agent.planning.risk import blocked_cells_from_belief, build_risk_map


class Planner:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.max_search_nodes = int(config.get("max_search_nodes", 20000))
        self.risk_threshold = float(config.get("risk_threshold", 0.65))
        self.goal_weight = float(config.get("planner_goal_weight", 4.0))
        self.info_weight = float(config.get("info_gain_weight", 1.5))
        self.ppo_weight = float(config.get("ppo_weight", 1.0))
        self.risk_weight = float(config.get("risk_weight", 3.0))
        self.step_cost_weight = float(config.get("step_cost_weight", 0.05))

    def plan_candidates(
        self,
        belief: BeliefState,
        graph: GraphState,
        candidates: list[CandidateSubgoal],
        ppo_scores: dict[str, float] | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> list[CandidateSubgoal]:
        if not candidates:
            return []
        start_obj = self._agent_object(belief, graph)
        if start_obj is None:
            return []
        start = (round(start_obj.centroid[0]), round(start_obj.centroid[1]))
        height, width = self._frame_shape(graph, frame_shape)
        blocked = blocked_cells_from_belief(belief, graph, self.risk_threshold)
        blocked.discard(start)
        risk = build_risk_map(belief, graph)
        planned: list[CandidateSubgoal] = []
        for candidate in candidates:
            if candidate.target_position is None:
                path = [int(GridAction.WAIT)]
            else:
                target = self._walkable_target(candidate.target_position, start, width, height, blocked)
                path = astar_path(start, target, width, height, blocked, risk, self.max_search_nodes)
                if path is None:
                    continue
            ppo_score = float((ppo_scores or {}).get(candidate.name, 0.0))
            score = self.score_candidate(candidate, path, ppo_score)
            planned.append(replace(candidate, planner_path=path, symbolic_score=score, ppo_score=ppo_score))
        return sorted(planned, key=lambda item: item.symbolic_score + self.ppo_weight * item.ppo_score, reverse=True)

    def choose(self, planned: list[CandidateSubgoal]) -> CandidateSubgoal | None:
        return planned[0] if planned else None

    def score_candidate(self, candidate: CandidateSubgoal, path: list[int], ppo_score: float = 0.0) -> float:
        return (
            self.goal_weight * candidate.expected_goal_progress
            + self.info_weight * candidate.expected_information_gain
            + self.ppo_weight * ppo_score
            - self.risk_weight * candidate.expected_risk
            - self.step_cost_weight * len(path)
        )

    def _agent_object(self, belief: BeliefState, graph: GraphState) -> ObjectComponent | None:
        best = None
        best_score = -1.0
        for node in graph.nodes:
            object_id = node.track_id or node.id
            score = belief.object_beliefs.get(object_id, {}).get("controllable", 0.0)
            if node.attributes.get("label") == "agent":
                return node
            if score > best_score:
                best = node
                best_score = score
        return best if best_score > 0 else (graph.nodes[0] if graph.nodes else None)

    def _frame_shape(self, graph: GraphState, frame_shape: tuple[int, int] | None) -> tuple[int, int]:
        if frame_shape is not None:
            return frame_shape
        if graph.graph_features is not None and len(graph.graph_features) >= 3:
            return int(graph.graph_features[2]), int(graph.graph_features[1])
        if not graph.nodes:
            return 1, 1
        width = max(node.bbox[2] for node in graph.nodes) + 1
        height = max(node.bbox[3] for node in graph.nodes) + 1
        return height, width

    def _walkable_target(
        self,
        target: tuple[int, int],
        start: tuple[int, int],
        width: int,
        height: int,
        blocked: set[tuple[int, int]],
    ) -> tuple[int, int]:
        if target not in blocked:
            return target
        options = [
            (target[0] + 1, target[1]),
            (target[0] - 1, target[1]),
            (target[0], target[1] + 1),
            (target[0], target[1] - 1),
        ]
        valid = [cell for cell in options if 0 <= cell[0] < width and 0 <= cell[1] < height and cell not in blocked]
        if not valid:
            return target
        return min(valid, key=lambda cell: abs(cell[0] - start[0]) + abs(cell[1] - start[1]))
