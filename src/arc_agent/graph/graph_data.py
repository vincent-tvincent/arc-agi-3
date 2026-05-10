"""Graph dataclasses used by perception, models, and planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arc_agent.perception.object_types import ObjectComponent


@dataclass
class GraphState:
    nodes: list[ObjectComponent]
    node_features: Any
    edge_index: Any
    edge_attr: Any
    graph_features: Any | None
    action_context: Any | None
    frame_index: int
    feature_names: list[str]
    edge_feature_names: list[str]

    def to_torch(self, device: str = "cpu") -> "GraphState":
        import torch

        return GraphState(
            nodes=self.nodes,
            node_features=torch.as_tensor(self.node_features, dtype=torch.float32, device=device),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            edge_attr=torch.as_tensor(self.edge_attr, dtype=torch.float32, device=device),
            graph_features=None
            if self.graph_features is None
            else torch.as_tensor(self.graph_features, dtype=torch.float32, device=device),
            action_context=None
            if self.action_context is None
            else torch.as_tensor(self.action_context, dtype=torch.float32, device=device),
            frame_index=self.frame_index,
            feature_names=self.feature_names,
            edge_feature_names=self.edge_feature_names,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_features.shape[1]) if self.node_features is not None and self.node_features.ndim == 2 else 0

    @property
    def edge_feature_dim(self) -> int:
        return int(self.edge_attr.shape[1]) if self.edge_attr is not None and self.edge_attr.ndim == 2 else 0


def empty_graph(frame_index: int, node_feature_dim: int, edge_feature_dim: int) -> GraphState:
    return GraphState(
        nodes=[],
        node_features=np.zeros((0, node_feature_dim), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_attr=np.zeros((0, edge_feature_dim), dtype=np.float32),
        graph_features=np.zeros((4,), dtype=np.float32),
        action_context=None,
        frame_index=frame_index,
        feature_names=[],
        edge_feature_names=[],
    )
