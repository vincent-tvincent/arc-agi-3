from __future__ import annotations

import json

import numpy as np

from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.perception.object_types import ObjectComponent
from arc_agent.training.gnn_dataset import ARCTransitionGNNDataset, batch_graph_examples, make_gnn_dataset


def test_arc_transition_dataset_reads_raw_official_rollout(tmp_path):
    path = tmp_path / "ls20_seed0.jsonl"
    row = {
        "step": 0,
        "action": "ACTION1",
        "before_hash": "before",
        "after_hash": "after",
        "before_frame": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "after_frame": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "changed_cells": [{"x": 1, "y": 1, "before": 1, "after": 0}],
        "state": "WIN",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = ARCTransitionGNNDataset(path, {})
    graph, labels = dataset[0]

    assert len(dataset) == 1
    assert graph.num_nodes == 1
    assert labels.shape == (1, len(AFFORDANCE_NAMES))
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("goal_related")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("collectible")] == 1.0


def test_make_gnn_dataset_auto_detects_training_examples(tmp_path):
    path = tmp_path / "ls20_seed0.examples.jsonl"
    row = {
        "step": 0,
        "action": {"name": "ACTION2", "data": {}},
        "input": {"frame": [[0, 2], [0, 0]]},
        "target": {
            "next_frame": [[0, 0], [0, 0]],
            "changed_cells": [{"x": 1, "y": 0, "before": 2, "after": 0}],
        },
        "outcome": {"frame_changed": True, "is_win": False, "is_game_over": False, "level_delta": 0},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = make_gnn_dataset(tmp_path, {}, "auto")

    assert isinstance(dataset, ARCTransitionGNNDataset)
    assert len(dataset) == 1


def test_batch_graph_examples_offsets_edges_and_concatenates_labels():
    graph_a = _graph(1, edge_index=np.array([[0], [1]], dtype=np.int64))
    graph_b = _graph(2, edge_index=np.array([[0], [1]], dtype=np.int64))
    labels_a = np.ones((2, len(AFFORDANCE_NAMES)), dtype=np.float32)
    labels_b = np.zeros((2, len(AFFORDANCE_NAMES)), dtype=np.float32)

    graph, labels = batch_graph_examples([(graph_a, labels_a), (graph_b, labels_b)])

    assert graph.num_nodes == 4
    assert graph.node_features.shape == (4, 3)
    assert graph.edge_index.tolist() == [[0, 2], [1, 3]]
    assert graph.edge_attr.shape == (2, 2)
    assert labels.shape == (4, len(AFFORDANCE_NAMES))
    assert labels[:2].sum() == 2 * len(AFFORDANCE_NAMES)
    assert labels[2:].sum() == 0.0


def _graph(frame_index: int, edge_index: np.ndarray) -> GraphState:
    nodes = [
        ObjectComponent(
            id=frame_index * 10,
            color_id=1,
            pixels=np.array([[0, 0]], dtype=np.int64),
            bbox=(0, 0, 0, 0),
            centroid=(0.0, 0.0),
            area=1,
            frame_index=frame_index,
        ),
        ObjectComponent(
            id=frame_index * 10 + 1,
            color_id=2,
            pixels=np.array([[1, 0]], dtype=np.int64),
            bbox=(1, 0, 1, 0),
            centroid=(1.0, 0.0),
            area=1,
            frame_index=frame_index,
        ),
    ]
    return GraphState(
        nodes=nodes,
        node_features=np.full((2, 3), frame_index, dtype=np.float32),
        edge_index=edge_index,
        edge_attr=np.full((edge_index.shape[1], 2), frame_index, dtype=np.float32),
        graph_features=np.array([frame_index], dtype=np.float32),
        action_context=None,
        frame_index=frame_index,
        feature_names=["a", "b", "c"],
        edge_feature_names=["d", "e"],
    )
