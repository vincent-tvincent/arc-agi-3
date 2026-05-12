"""Build object-centric graph states from tracked components."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np

from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES, EDGE_FEATURE_NAMES, node_feature_names
from arc_agent.perception.object_types import ObjectComponent


@dataclass
class GraphBuilderConfig:
    spatial_radius: float = 6.0
    max_colors: int = 16
    action_space_n: int = 8
    include_touching_edges: bool = True
    include_same_color_edges: bool = True
    include_same_row_col_edges: bool = True
    include_temporal_edges: bool = True
    include_candidate_causal_edges: bool = True


class GraphBuilder:
    def __init__(self, config: dict[str, Any] | None = None, feature_config: dict[str, Any] | None = None) -> None:
        config = config or {}
        feature_config = feature_config or {}
        self.config = GraphBuilderConfig(
            spatial_radius=float(config.get("spatial_radius", 6)),
            max_colors=int(feature_config.get("max_colors", 16)),
            action_space_n=int(feature_config.get("action_space_n", 8)),
            include_touching_edges=bool(config.get("include_touching_edges", True)),
            include_same_color_edges=bool(config.get("include_same_color_edges", True)),
            include_same_row_col_edges=bool(config.get("include_same_row_col_edges", True)),
            include_temporal_edges=bool(config.get("include_temporal_edges", True)),
            include_candidate_causal_edges=bool(config.get("include_candidate_causal_edges", True)),
        )
        self.feature_names = node_feature_names(self.config.max_colors, self.config.action_space_n, AFFORDANCE_NAMES)
        self.edge_feature_names = list(EDGE_FEATURE_NAMES)

    def build(
        self,
        tracked_objects: list[ObjectComponent],
        previous_action: int | None = None,
        previous_graph: GraphState | None = None,
        belief: Any | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> GraphState:
        height, width = self._frame_shape(tracked_objects, frame_shape)
        agent = self._agent_candidate(tracked_objects, belief)
        previous_by_track = {
            obj.track_id: obj for obj in (previous_graph.nodes if previous_graph is not None else []) if obj.track_id is not None
        }
        nodes = list(tracked_objects)
        node_features = np.asarray(
            [self._node_features(obj, width, height, previous_by_track.get(obj.track_id), previous_action, belief, agent) for obj in nodes],
            dtype=np.float32,
        )
        if not nodes:
            node_features = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        edge_index, edge_attr = self._edges(nodes, belief)
        graph_features = np.asarray([len(nodes), width, height, 0 if previous_action is None else previous_action], dtype=np.float32)
        action_context = self._action_context(previous_action)
        frame_index = nodes[0].frame_index if nodes else 0
        return GraphState(
            nodes=nodes,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_features=graph_features,
            action_context=action_context,
            frame_index=frame_index,
            feature_names=self.feature_names,
            edge_feature_names=self.edge_feature_names,
        )

    def _node_features(
        self,
        obj: ObjectComponent,
        width: int,
        height: int,
        previous: ObjectComponent | None,
        previous_action: int | None,
        belief: Any | None,
        agent: ObjectComponent | None,
    ) -> list[float]:
        max_colors = self.config.max_colors
        norm_w = max(width - 1, 1)
        norm_h = max(height - 1, 1)
        features: list[float] = [
            float(obj.centroid[0] / norm_w),
            float(obj.centroid[1] / norm_h),
            float(obj.width / max(width, 1)),
            float(obj.height / max(height, 1)),
            float(obj.area / max(width * height, 1)),
        ]
        color = obj.color_id if obj.color_id is not None else -1
        features.extend(1.0 if color == idx else 0.0 for idx in range(max_colors))
        if previous is not None:
            dx = (obj.centroid[0] - previous.centroid[0]) / max(width, 1)
            dy = (obj.centroid[1] - previous.centroid[1]) / max(height, 1)
        else:
            dx = float(obj.attributes.get("dx", 0.0)) / max(width, 1)
            dy = float(obj.attributes.get("dy", 0.0)) / max(height, 1)
        changed = 1.0 if obj.attributes.get("changed") else 0.0
        is_new = 1.0 if obj.attributes.get("is_new") else 0.0
        disappeared = 1.0 if obj.attributes.get("disappeared") else 0.0
        dist_to_agent = 0.0
        touching_agent = 0.0
        if agent is not None and agent is not obj:
            dist = hypot(obj.centroid[0] - agent.centroid[0], obj.centroid[1] - agent.centroid[1])
            dist_to_agent = float(dist / max(width + height, 1))
            touching_agent = 1.0 if touching(obj, agent) else 0.0
        features.extend([dx, dy, changed, is_new, disappeared, dist_to_agent, touching_agent])
        beliefs = getattr(belief, "object_beliefs", {}) if belief is not None else {}
        obj_belief = beliefs.get(obj.track_id or obj.id, {})
        features.extend(float(obj_belief.get(name, 0.1)) for name in AFFORDANCE_NAMES)
        action = self._action_context(previous_action)
        features.extend(action.tolist())
        return features

    def _edges(self, nodes: list[ObjectComponent], belief: Any | None) -> tuple[np.ndarray, np.ndarray]:
        indices: list[tuple[int, int]] = []
        attrs: list[list[float]] = []
        causal = getattr(belief, "causal_edges", {}) if belief is not None else {}
        reachable = getattr(belief, "reachable_cells", set()) if belief is not None else set()
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i == j:
                    continue
                attr = self._edge_features(source, target, causal, reachable)
                if self._include_edge(attr):
                    indices.append((i, j))
                    attrs.append(attr)
        if not indices:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, len(self.edge_feature_names)), dtype=np.float32)
        edge_index = np.asarray(indices, dtype=np.int64).T
        edge_attr = np.asarray(attrs, dtype=np.float32)
        return edge_index, edge_attr

    def _edge_features(
        self,
        source: ObjectComponent,
        target: ObjectComponent,
        causal: dict[tuple[int, int], dict[str, float]],
        reachable: set[tuple[int, int]],
    ) -> list[float]:
        dx = target.centroid[0] - source.centroid[0]
        dy = target.centroid[1] - source.centroid[1]
        distance = hypot(dx, dy)
        source_id = source.track_id or source.id
        target_id = target.track_id or target.id
        causal_probability = float(causal.get((source_id, target_id), {}).get("probability", 0.0))
        target_reachable = 1.0 if (round(target.centroid[0]), round(target.centroid[1])) in reachable else 0.0
        return [
            float(dx),
            float(dy),
            float(distance),
            1.0 if touching(source, target) else 0.0,
            1.0 if source.color_id == target.color_id else 0.0,
            1.0 if source.bbox[1] <= target.centroid[1] <= source.bbox[3] else 0.0,
            1.0 if source.bbox[0] <= target.centroid[0] <= source.bbox[2] else 0.0,
            1.0 if source.track_id is not None and source.track_id == target.track_id else 0.0,
            causal_probability,
            target_reachable,
        ]

    def _include_edge(self, attr: list[float]) -> bool:
        distance = attr[2]
        return (
            distance <= self.config.spatial_radius
            or (self.config.include_touching_edges and attr[3] > 0)
            or (self.config.include_same_color_edges and attr[4] > 0)
            or (self.config.include_same_row_col_edges and (attr[5] > 0 or attr[6] > 0))
            or (self.config.include_candidate_causal_edges and attr[8] > 0)
        )

    def _action_context(self, previous_action: int | None) -> np.ndarray:
        context = np.zeros((self.config.action_space_n,), dtype=np.float32)
        if previous_action is not None and 0 <= int(previous_action) < self.config.action_space_n:
            context[int(previous_action)] = 1.0
        return context

    def _frame_shape(self, objects: list[ObjectComponent], frame_shape: tuple[int, ...] | None) -> tuple[int, int]:
        if frame_shape is not None:
            if len(frame_shape) < 2:
                return (1, 1)
            return int(frame_shape[0]), int(frame_shape[1])
        if not objects:
            return (1, 1)
        width = max(obj.bbox[2] for obj in objects) + 1
        height = max(obj.bbox[3] for obj in objects) + 1
        return height, width

    def _agent_candidate(self, objects: list[ObjectComponent], belief: Any | None) -> ObjectComponent | None:
        if not objects:
            return None
        beliefs = getattr(belief, "object_beliefs", {}) if belief is not None else {}
        best_obj = None
        best_score = -1.0
        for obj in objects:
            if obj.attributes.get("label") == "agent":
                return obj
            score = float(beliefs.get(obj.track_id or obj.id, {}).get("controllable", 0.0))
            if score > best_score:
                best_obj = obj
                best_score = score
        return best_obj if best_score > 0 else objects[0]


def touching(a: ObjectComponent, b: ObjectComponent) -> bool:
    if a.bbox[2] + 1 < b.bbox[0] or b.bbox[2] + 1 < a.bbox[0]:
        return False
    if a.bbox[3] + 1 < b.bbox[1] or b.bbox[3] + 1 < a.bbox[1]:
        return False
    return True
