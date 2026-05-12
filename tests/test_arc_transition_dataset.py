from __future__ import annotations

import json

import numpy as np
import pytest

from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.perception.object_types import ObjectComponent
from arc_agent.training.gnn_dataset import (
    ARCTransitionGNNDataset,
    CachedGNNDataset,
    batch_graph_examples,
    graph_cache_config_hash,
    graph_to_cache_dict,
    make_gnn_dataset,
)


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
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0
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


def test_arc_transition_dataset_reads_gt_recording_json_pairs(tmp_path):
    path = tmp_path / "ls20-recording.json"
    rows = [
        {
            "timestamp": "2026-03-02T00:00:00+00:00",
            "data": {
                "game_id": "ls20-9607627b",
                "guid": "episode-1",
                "frame": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "action_input": {"id": 0, "data": {}},
                "available_actions": [1, 2],
            },
        },
        {
            "timestamp": "2026-03-02T00:00:01+00:00",
            "data": {
                "game_id": "ls20-9607627b",
                "guid": "episode-1",
                "frame": [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                "state": "WIN",
                "levels_completed": 1,
                "action_input": {"id": 1, "data": {}},
                "available_actions": [1, 2],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = ARCTransitionGNNDataset(path, {})
    graph, labels = dataset[0]

    assert len(dataset) == 1
    assert graph.num_nodes == 1
    assert graph.frame_index == 0
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("goal_related")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("collectible")] == 1.0


def test_pseudo_labels_do_not_use_global_color_change_as_interaction(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((6, 6), dtype=np.int64)
    after = before.copy()
    before[1, 1] = 2
    before[4, 4] = 2
    after[1, 1] = 0
    passive_same_color = component(1, 2, [(4, 4)])

    labels = dataset._pseudo_labels(
        [passive_same_color],
        before,
        after,
        {"step": 0, "changed_cells": [{"x": 1, "y": 1}]},
    )

    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("collectible")] == 0.0


def test_pseudo_labels_mark_status_bar_as_ui_not_interactable(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((10, 20), dtype=np.int64)
    after = before.copy()
    before[9, 2:18] = 11
    after[9, 2:10] = 0
    bar = component(1, 11, [(x, 9) for x in range(2, 18)])

    labels = dataset._pseudo_labels(
        [bar],
        before,
        after,
        {"step": 0, "changed_cells": [{"x": x, "y": 9} for x in range(10, 18)]},
    )

    assert labels[0, AFFORDANCE_NAMES.index("ui_or_status")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("movable")] == 0.0


def test_pseudo_labels_mark_bottom_hud_as_ui_not_unknown(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((20, 20), dtype=np.int64)
    after = before.copy()
    hud_cells = [(x, y) for y in range(15, 18) for x in range(1, 4)]
    for x, y in hud_cells:
        before[y, x] = 5
    after[16, 2] = 0
    hud = component(1, 5, hud_cells)

    labels = dataset._pseudo_labels(
        [hud],
        before,
        after,
        {"step": 0, "changed_cells": [{"x": 2, "y": 16}]},
    )

    assert labels[0, AFFORDANCE_NAMES.index("ui_or_status")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("trigger")] == 0.0


def test_pseudo_labels_mark_preview_shape_inside_hud_container_as_ui(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((64, 64), dtype=np.int64)
    after = before.copy()
    panel_cells = []
    for x in range(2, 17):
        panel_cells.append((x, 30))
        panel_cells.append((x, 46))
    for y in range(30, 47):
        panel_cells.append((2, y))
        panel_cells.append((16, y))
    shape_cells = [(5, 34), (6, 34), (7, 34), (5, 35), (5, 36), (5, 37), (9, 38), (10, 38)]
    shifted_shape_cells = [(x + 1, y) for x, y in shape_cells]
    for x, y in panel_cells:
        before[y, x] = 5
        after[y, x] = 5
    for x, y in shape_cells:
        before[y, x] = 9
    for x, y in shifted_shape_cells:
        after[y, x] = 9
    panel = component(1, 5, panel_cells)
    preview_shape = component(2, 9, shape_cells)
    changed = [{"x": x, "y": y} for x, y in shape_cells + shifted_shape_cells]

    labels = dataset._pseudo_labels(
        [panel, preview_shape],
        before,
        after,
        {"step": 0, "changed_cells": changed},
    )

    assert labels[1, AFFORDANCE_NAMES.index("ui_or_status")] == 1.0
    assert labels[1, AFFORDANCE_NAMES.index("movable")] == 0.0
    assert labels[1, AFFORDANCE_NAMES.index("controllable")] == 0.0
    assert labels[1, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0


def test_pseudo_labels_do_not_use_center_container_as_hud_preview(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((64, 64), dtype=np.int64)
    after = before.copy()
    panel_cells = []
    for x in range(25, 40):
        panel_cells.append((x, 25))
        panel_cells.append((x, 40))
    for y in range(25, 41):
        panel_cells.append((25, y))
        panel_cells.append((39, y))
    shape_cells = [(29, 30), (30, 30), (31, 30), (29, 31), (29, 32), (29, 33)]
    shifted_shape_cells = [(x + 1, y) for x, y in shape_cells]
    for x, y in panel_cells:
        before[y, x] = 5
        after[y, x] = 5
    for x, y in shape_cells:
        before[y, x] = 9
    for x, y in shifted_shape_cells:
        after[y, x] = 9
    panel = component(1, 5, panel_cells)
    inner_shape = component(2, 9, shape_cells)
    changed = [{"x": x, "y": y} for x, y in shape_cells + shifted_shape_cells]

    labels = dataset._pseudo_labels(
        [panel, inner_shape],
        before,
        after,
        {"step": 0, "changed_cells": changed},
    )

    assert labels[1, AFFORDANCE_NAMES.index("ui_or_status")] == 0.0
    assert labels[1, AFFORDANCE_NAMES.index("movable")] == 1.0


def test_pseudo_labels_suppress_large_floor_like_regions(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((10, 10), dtype=np.int64)
    after = before.copy()
    floor_cells = [(x, y) for y in range(2, 8) for x in range(2, 8)]
    for x, y in floor_cells:
        before[y, x] = 3
        after[y, x] = 3
    after[4, 4] = 4
    floor = component(1, 3, floor_cells)

    labels = dataset._pseudo_labels(
        [floor],
        before,
        after,
        {"step": 0, "changed_cells": [{"x": 4, "y": 4}]},
    )

    assert labels[0].sum() == 0.0


def test_pseudo_label_thresholds_are_configurable(tmp_path):
    dataset = ARCTransitionGNNDataset(
        _empty_jsonl(tmp_path),
        {"gnn_training": {"pseudo_labeling": {"floor_area_ratio": 0.5}}},
    )
    before = np.zeros((10, 10), dtype=np.int64)
    after = before.copy()
    floor_cells = [(x, y) for y in range(2, 8) for x in range(2, 8)]
    for x, y in floor_cells:
        before[y, x] = 3
        after[y, x] = 3
    after[4, 4] = 4
    floor = component(1, 3, floor_cells)

    labels = dataset._pseudo_labels(
        [floor],
        before,
        after,
        {"step": 0, "changed_cells": [{"x": 4, "y": 4}]},
    )

    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 1.0


def test_pseudo_labels_keep_coherent_motion_as_movable(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((8, 8), dtype=np.int64)
    after = before.copy()
    before[2, 2] = 4
    after[2, 3] = 4
    moving = component(1, 4, [(2, 2)])

    labels = dataset._pseudo_labels(
        [moving],
        before,
        after,
        {"step": 0, "action": "ACTION1", "changed_cells": [{"x": 2, "y": 2}, {"x": 3, "y": 2}]},
    )

    assert labels[0, AFFORDANCE_NAMES.index("movable")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("controllable")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("collectible")] == 0.0


def test_soft_labels_scale_square_agent_by_motion_and_size(tmp_path):
    dataset = ARCTransitionGNNDataset(
        _empty_jsonl(tmp_path),
        {
            "gnn_training": {
                "pseudo_labeling": {
                    "soft_labels": {
                        "enabled": True,
                        "short_motion_movable": 0.9,
                        "short_motion_controllable": 0.85,
                    },
                    "controllable_max_step_distance": 2.0,
                    "motion_score_floor": 0.5,
                    "controllable_max_area": 64,
                    "controllable_max_area_ratio": 0.0,
                    "controllable_smallness_floor": 0.25,
                }
            }
        },
    )
    before = np.zeros((12, 12), dtype=np.int64)
    after = before.copy()
    agent_cells = [(4, 4), (5, 4), (6, 4), (4, 5), (5, 5), (6, 5), (4, 6), (5, 6), (6, 6)]
    moved_cells = [(x + 1, y) for x, y in agent_cells]
    for x, y in agent_cells:
        before[y, x] = 4
    for x, y in moved_cells:
        after[y, x] = 4
    agent = component(1, 4, agent_cells)
    changed = [{"x": x, "y": y} for x, y in agent_cells + moved_cells]

    labels = dataset._pseudo_labels(
        [agent],
        before,
        after,
        {"step": 0, "action": "ACTION1", "changed_cells": changed},
    )

    assert 0.0 < labels[0, AFFORDANCE_NAMES.index("movable")] < 1.0
    assert 0.0 < labels[0, AFFORDANCE_NAMES.index("controllable")] < labels[0, AFFORDANCE_NAMES.index("movable")]
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0


def test_soft_labels_keep_hud_preview_residual_motion(tmp_path):
    dataset = ARCTransitionGNNDataset(
        _empty_jsonl(tmp_path),
        {
            "gnn_training": {
                "pseudo_labeling": {
                    "soft_labels": {
                        "enabled": True,
                        "hud_preview_ui": 0.9,
                        "hud_preview_movable_residual": 0.1,
                        "hud_preview_unknown_residual": 0.05,
                    }
                }
            }
        },
    )
    before = np.zeros((64, 64), dtype=np.int64)
    after = before.copy()
    panel_cells = []
    for x in range(2, 17):
        panel_cells.append((x, 30))
        panel_cells.append((x, 46))
    for y in range(30, 47):
        panel_cells.append((2, y))
        panel_cells.append((16, y))
    shape_cells = [(5, 34), (6, 34), (7, 34), (5, 35), (5, 36), (5, 37)]
    shifted_shape_cells = [(x + 1, y) for x, y in shape_cells]
    for x, y in panel_cells:
        before[y, x] = 5
        after[y, x] = 5
    for x, y in shape_cells:
        before[y, x] = 9
    for x, y in shifted_shape_cells:
        after[y, x] = 9
    preview_shape = component(2, 9, shape_cells)
    panel = component(1, 5, panel_cells)
    changed = [{"x": x, "y": y} for x, y in shape_cells + shifted_shape_cells]

    labels = dataset._pseudo_labels(
        [panel, preview_shape],
        before,
        after,
        {"step": 0, "changed_cells": changed},
    )

    assert labels[1, AFFORDANCE_NAMES.index("ui_or_status")] == 0.9
    assert labels[1, AFFORDANCE_NAMES.index("movable")] == 0.1
    assert labels[1, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.05


def test_pseudo_labels_mark_stationary_remote_control_as_trigger_not_unknown(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((12, 12), dtype=np.int64)
    after = before.copy()
    before[3, 3] = 6
    after[3, 3] = 6
    after[8, 8] = 2
    button = component(1, 6, [(3, 3)])

    labels = dataset._pseudo_labels(
        [button],
        before,
        after,
        {"step": 0, "changed_cells": [{"x": 3, "y": 3}, {"x": 8, "y": 8}]},
    )

    assert labels[0, AFFORDANCE_NAMES.index("trigger")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0


def test_pseudo_labels_mark_remote_target_shift_as_goal_not_trigger(tmp_path):
    dataset = ARCTransitionGNNDataset(_empty_jsonl(tmp_path), {})
    before = np.zeros((12, 12), dtype=np.int64)
    after = before.copy()
    target_cells = [(5, 8), (6, 8), (5, 9), (6, 9)]
    shifted_cells = [(5, 3), (6, 3), (5, 4), (6, 4)]
    for x, y in target_cells:
        before[y, x] = 7
    for x, y in shifted_cells:
        after[y, x] = 7
    target = component(1, 7, target_cells)
    changed = [{"x": x, "y": y} for x, y in target_cells + shifted_cells]

    labels = dataset._pseudo_labels(
        [target],
        before,
        after,
        {"step": 0, "changed_cells": changed},
    )

    assert labels[0, AFFORDANCE_NAMES.index("goal_related")] == 1.0
    assert labels[0, AFFORDANCE_NAMES.index("trigger")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("controllable")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("movable")] == 0.0
    assert labels[0, AFFORDANCE_NAMES.index("unknown_interactable")] == 0.0


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


def test_cached_gnn_dataset_round_trips_graph_and_labels(tmp_path):
    torch = pytest.importorskip("torch")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    graph = _graph(3, edge_index=np.array([[0], [1]], dtype=np.int64))
    labels = np.full((2, len(AFFORDANCE_NAMES)), 0.25, dtype=np.float32)
    torch.save(
        [{"source_index": 0, "graph": graph_to_cache_dict(graph), "labels": labels}],
        cache_dir / "shard_000000.pt",
    )
    config: dict = {}
    manifest = {
        "format": "arc_agent_gnn_cache",
        "version": 1,
        "labeler_config_hash": graph_cache_config_hash(config),
        "num_examples": 1,
        "num_shards": 1,
        "affordances": AFFORDANCE_NAMES,
        "entries": [{"shard": "shard_000000.pt", "index": 0}],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = make_gnn_dataset(cache_dir, config, "cache")
    loaded_graph, loaded_labels = dataset[0]

    assert isinstance(dataset, CachedGNNDataset)
    assert loaded_graph.num_nodes == graph.num_nodes
    assert loaded_graph.edge_index.tolist() == graph.edge_index.tolist()
    np.testing.assert_allclose(loaded_labels, labels)


def _empty_jsonl(tmp_path):
    path = tmp_path / "empty.examples.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def component(obj_id: int, color: int, cells: list[tuple[int, int]]) -> ObjectComponent:
    pixels = np.asarray(cells, dtype=np.int64)
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return ObjectComponent(
        id=obj_id,
        color_id=color,
        pixels=pixels,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
        area=len(cells),
        frame_index=0,
    )


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
