"""End-to-end object-centric agent loop."""

from __future__ import annotations

from typing import Any

from arc_agent.belief.belief_state import BeliefState
from arc_agent.belief.belief_update import BeliefUpdater, compare_graphs
from arc_agent.execution.action_executor import ActionExecutor
from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.perception.object_tracker import ObjectTracker
from arc_agent.perception.segmenter import GridSegmenter
from arc_agent.planning.candidates import CandidateGenerator
from arc_agent.planning.planner import Planner


class ObjectCentricAgent:
    """Pipeline agent: perception -> graph -> belief -> candidates -> planner -> action."""

    def __init__(self, config: dict[str, Any], color_label_map: dict[int, set[str]] | None = None) -> None:
        self.config = config
        self.segmenter = GridSegmenter(config.get("perception", {}))
        self.tracker = ObjectTracker(config.get("tracking", {}))
        self.graph_builder = GraphBuilder(config.get("edge_builder", {}), config.get("features", {}))
        self.belief = BeliefState()
        self.belief_updater = BeliefUpdater(config.get("belief", {}))
        self.candidates = CandidateGenerator(config.get("candidate_generator", {}))
        self.planner = Planner(config.get("planner", {}))
        self.executor = ActionExecutor()
        self.color_label_map = color_label_map or {}
        self.previous_graph: GraphState | None = None
        self.last_action: int | None = None
        self.last_reward = 0.0
        self.last_done = False
        self.frame_index = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.belief = BeliefState()
        self.previous_graph = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_done = False
        self.frame_index = 0

    def observe_outcome(self, reward: float, done: bool) -> None:
        self.last_reward = reward
        self.last_done = done

    def act(self, obs: Any) -> int:
        objects = self.segmenter.segment(obs, frame_index=self.frame_index)
        for obj in objects:
            labels = self.color_label_map.get(int(obj.color_id or -1), set())
            if labels:
                obj.attributes["label"] = next(iter(labels))
        tracked = self.tracker.update(objects)
        graph = self.graph_builder.build(
            tracked_objects=tracked.objects,
            previous_action=self.last_action,
            previous_graph=self.previous_graph,
            belief=self.belief,
            frame_shape=getattr(obs, "shape", None),
        )
        self._seed_mock_beliefs(graph)
        delta = compare_graphs(self.previous_graph, graph, self.last_action, self.last_reward, self.last_done)
        self.belief_updater.update(self.belief, graph, None, delta, self.last_reward, self.last_done)
        candidates = self.candidates.generate(self.belief, graph, None)
        planned = self.planner.plan_candidates(self.belief, graph, candidates, frame_shape=getattr(obs, "shape", None))
        chosen = self.planner.choose(planned)
        action = self.executor.next_action(chosen)
        self.previous_graph = graph
        self.last_action = action
        self.frame_index += 1
        return action

    def _seed_mock_beliefs(self, graph: GraphState) -> None:
        """Use synthetic labels only when the caller supplies them."""
        if not self.color_label_map:
            return
        for node in graph.nodes:
            object_id = node.track_id or node.id
            belief = self.belief.ensure_object(object_id)
            labels = self.color_label_map.get(int(node.color_id or -1), set())
            if "controllable" in labels:
                node.attributes["label"] = "agent"
            for name in AFFORDANCE_NAMES:
                if name in labels:
                    belief[name] = max(belief.get(name, 0.1), 0.9)
            if "blocking" in labels:
                belief["blocking"] = 0.9
            if "goal_related" in labels:
                self.belief.goals.add(object_id)
            if "hazardous" in labels:
                self.belief.hazards.add(object_id)
