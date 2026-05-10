"""Graph datasets for affordance pretraining."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from arc_agent.envs.mock_grid_env import COLOR_LABELS
from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.perception.object_tracker import ObjectTracker
from arc_agent.perception.object_types import ObjectComponent
from arc_agent.perception.segmenter import GridSegmenter


class MockGNNDataset:
    def __init__(self, path: str | Path, config: dict[str, Any]) -> None:
        self.path = Path(path)
        self.config = config
        self.rows = self._read_rows()
        self.segmenter = GridSegmenter(config.get("perception", {}))
        self.builder = GraphBuilder(config.get("edge_builder", {}), config.get("features", {}))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, np.ndarray]:
        row = self.rows[index]
        tracker = ObjectTracker(self.config.get("tracking", {}))
        objects = tracker.update(self.segmenter.segment(row["frame"], frame_index=int(row.get("step", 0)))).objects
        for obj in objects:
            labels = COLOR_LABELS.get(int(obj.color_id or -1), set())
            if labels:
                obj.attributes["label"] = "agent" if "controllable" in labels else next(iter(labels))
        graph = self.builder.build(objects, previous_action=row.get("action"), frame_shape=np.asarray(row["frame"]).shape)
        labels = np.zeros((len(graph.nodes), len(AFFORDANCE_NAMES)), dtype=np.float32)
        for node_idx, node in enumerate(graph.nodes):
            color_labels = COLOR_LABELS.get(int(node.color_id or -1), set())
            for label in color_labels:
                if label in AFFORDANCE_NAMES:
                    labels[node_idx, AFFORDANCE_NAMES.index(label)] = 1.0
            if "blocking" in color_labels:
                labels[node_idx, AFFORDANCE_NAMES.index("blocking")] = 1.0
        return graph, labels

    def split(self, validation_fraction: float) -> tuple["DatasetView", "DatasetView"]:
        split_idx = int(len(self) * (1.0 - validation_fraction))
        return DatasetView(self, list(range(split_idx))), DatasetView(self, list(range(split_idx, len(self))))

    def _read_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.path.is_dir():
            paths = sorted(self.path.glob("*.jsonl"))
        else:
            paths = [self.path]
        for path in paths:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        rows.append(json.loads(line))
        return rows


class DatasetView:
    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, np.ndarray]:
        return self.dataset[self.indices[index]]


class ARCTransitionGNNDataset:
    """Build graph examples from official rollout/training-example JSONL.

    Official games do not expose affordance labels. This dataset derives noisy
    pseudo-labels from transitions so the GNN can be warmed up on real frames.
    """

    def __init__(
        self,
        path: str | Path,
        config: dict[str, Any],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.config = config
        train_cfg = config.get("gnn_training", {})
        self.lazy_index = bool(train_cfg.get("lazy_index", True))
        self.index_workers = max(1, int(train_cfg.get("index_workers", 1)))
        self.rows: list[dict[str, Any]] = []
        self.entries: list[tuple[Path, int]] = []
        if self.lazy_index:
            self.entries = self._index_rows(progress_callback)
        else:
            self.rows = [row for row in self._read_rows(progress_callback) if self._normalize_row(row) is not None]
        self.segmenter = GridSegmenter(config.get("perception", {}))
        self.builder = GraphBuilder(config.get("edge_builder", {}), config.get("features", {}))

    def __len__(self) -> int:
        return len(self.entries) if self.lazy_index else len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, np.ndarray]:
        row = self._read_entry(index) if self.lazy_index else self.rows[index]
        normalized = self._normalize_row(row)
        if normalized is None:
            raise IndexError(index)
        before = np.asarray(normalized["before_frame"], dtype=np.int64)
        after = np.asarray(normalized["after_frame"], dtype=np.int64)
        tracker = ObjectTracker(self.config.get("tracking", {}))
        objects = tracker.update(self.segmenter.segment(before, frame_index=int(normalized["step"]))).objects
        graph = self.builder.build(
            objects,
            previous_action=self._action_index(normalized["action"]),
            frame_shape=before.shape,
        )
        labels = self._pseudo_labels(graph.nodes, before, after, normalized)
        return graph, labels

    def split(self, validation_fraction: float) -> tuple[DatasetView, DatasetView]:
        split_idx = int(len(self) * (1.0 - validation_fraction))
        return DatasetView(self, list(range(split_idx))), DatasetView(self, list(range(split_idx, len(self))))

    def _pseudo_labels(
        self,
        objects: list[ObjectComponent],
        before: np.ndarray,
        after: np.ndarray,
        row: dict[str, Any],
    ) -> np.ndarray:
        labels = np.zeros((len(objects), len(AFFORDANCE_NAMES)), dtype=np.float32)
        changed_cells = {(int(cell["x"]), int(cell["y"])) for cell in row.get("changed_cells", [])}
        changed_colors_before = {
            int(before[y, x])
            for x, y in changed_cells
            if 0 <= y < before.shape[0] and 0 <= x < before.shape[1]
        }
        changed_colors_after = {
            int(after[y, x])
            for x, y in changed_cells
            if 0 <= y < after.shape[0] and 0 <= x < after.shape[1]
        }
        after_components = self.segmenter.segment(after, frame_index=int(row["step"]) + 1)
        after_by_color: dict[int, list[ObjectComponent]] = {}
        for component in after_components:
            if component.color_id is not None:
                after_by_color.setdefault(int(component.color_id), []).append(component)

        for node_idx, node in enumerate(objects):
            object_cells = node.cells
            object_id = node.track_id or node.id
            del object_id
            intersects_change = bool(object_cells & changed_cells)
            adjacent_change = any(abs(x - cx) + abs(y - cy) <= 1 for x, y in object_cells for cx, cy in changed_cells)
            color_changed = node.color_id in changed_colors_before or node.color_id in changed_colors_after
            if intersects_change or adjacent_change or color_changed:
                labels[node_idx, AFFORDANCE_NAMES.index("unknown_interactable")] = 1.0
            if row.get("is_win") or int(row.get("level_delta", 0)) > 0:
                if intersects_change or adjacent_change or color_changed:
                    labels[node_idx, AFFORDANCE_NAMES.index("goal_related")] = 1.0
            if row.get("is_game_over"):
                if intersects_change or adjacent_change or color_changed:
                    labels[node_idx, AFFORDANCE_NAMES.index("hazardous")] = 1.0
            if node.color_id is not None:
                after_same_color = after_by_color.get(int(node.color_id), [])
                if not any(bool(node.cells & component.cells) for component in after_same_color):
                    if intersects_change or color_changed:
                        labels[node_idx, AFFORDANCE_NAMES.index("collectible")] = 1.0
                if any(
                    component.centroid != node.centroid and _centroid_distance(component, node) <= 4.0
                    for component in after_same_color
                ):
                    labels[node_idx, AFFORDANCE_NAMES.index("movable")] = 1.0
            if not row.get("frame_changed", True) and adjacent_change:
                labels[node_idx, AFFORDANCE_NAMES.index("blocking")] = 1.0
        return labels

    def _index_rows(self, progress_callback: Callable[[str], None] | None = None) -> list[tuple[Path, int]]:
        if self.path.is_dir():
            paths = sorted(self.path.glob("*.jsonl"))
        else:
            paths = [self.path]
        if self.index_workers <= 1 or len(paths) <= 1:
            return self._index_rows_serial(paths, progress_callback)

        entries_by_path: dict[Path, list[tuple[Path, int]]] = {}
        total_files = len(paths)
        completed = 0
        total_rows = 0
        with ThreadPoolExecutor(max_workers=self.index_workers) as executor:
            futures = {executor.submit(_index_jsonl_file, path): path for path in paths}
            for future in as_completed(futures):
                path = futures[future]
                file_entries = future.result()
                entries_by_path[path] = file_entries
                completed += 1
                total_rows += len(file_entries)
                if progress_callback is not None and (completed == 1 or completed % 100 == 0 or completed == total_files):
                    progress_callback(
                        f"indexed ARC jsonl files={completed}/{total_files} rows={total_rows} workers={self.index_workers}"
                    )
        entries: list[tuple[Path, int]] = []
        for path in paths:
            entries.extend(entries_by_path.get(path, []))
        return entries

    def _index_rows_serial(
        self,
        paths: list[Path],
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[tuple[Path, int]]:
        entries: list[tuple[Path, int]] = []
        total_files = len(paths)
        for file_idx, path in enumerate(paths, start=1):
            entries.extend(_index_jsonl_file(path))
            if progress_callback is not None and (file_idx == 1 or file_idx % 100 == 0 or file_idx == total_files):
                progress_callback(f"indexed ARC jsonl files={file_idx}/{total_files} rows={len(entries)} workers=1")
        return entries

    def _read_entry(self, index: int) -> dict[str, Any]:
        path, offset = self.entries[index]
        with path.open("r", encoding="utf-8") as file:
            file.seek(offset)
            return json.loads(file.readline())

    def _read_rows(self, progress_callback: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.path.is_dir():
            paths = sorted(self.path.glob("*.jsonl"))
        else:
            paths = [self.path]
        total_files = len(paths)
        for file_idx, path in enumerate(paths, start=1):
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        rows.append(json.loads(line))
            if progress_callback is not None and (file_idx == 1 or file_idx % 100 == 0 or file_idx == total_files):
                progress_callback(f"loaded ARC jsonl files={file_idx}/{total_files} rows={len(rows)} lazy_index=false")
        return rows

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if "before_frame" in row and "after_frame" in row:
            return {
                "step": int(row.get("step", 0)),
                "before_frame": row["before_frame"],
                "after_frame": row["after_frame"],
                "changed_cells": row.get("changed_cells", []),
                "action": row.get("action"),
                "frame_changed": row.get("before_hash") != row.get("after_hash"),
                "is_win": row.get("state") == "WIN",
                "is_game_over": row.get("state") == "GAME_OVER",
                "level_delta": 0,
            }
        if "input" in row and "target" in row:
            outcome = row.get("outcome", {})
            return {
                "step": int(row.get("step", 0)),
                "before_frame": row["input"].get("frame"),
                "after_frame": row["target"].get("next_frame"),
                "changed_cells": row["target"].get("changed_cells", []),
                "action": row.get("action", {}).get("name"),
                "frame_changed": bool(outcome.get("frame_changed", False)),
                "is_win": bool(outcome.get("is_win", False)),
                "is_game_over": bool(outcome.get("is_game_over", False)),
                "level_delta": int(outcome.get("level_delta", 0)),
            }
        return None

    def _action_index(self, action: Any) -> int | None:
        if action is None:
            return None
        if isinstance(action, int):
            return action
        text = str(action).upper()
        if text.startswith("ACTION"):
            suffix = text.removeprefix("ACTION")
            if suffix.isdigit():
                return max(0, int(suffix) - 1)
        return None


def make_gnn_dataset(
    path: str | Path,
    config: dict[str, Any],
    data_format: str = "auto",
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    data_format = data_format.lower()
    if data_format == "mock":
        return MockGNNDataset(path, config)
    if data_format in {"arc", "arc_transitions", "official"}:
        return ARCTransitionGNNDataset(path, config, progress_callback)
    probe_path = Path(path)
    sample: dict[str, Any] | None = None
    paths = sorted(probe_path.glob("*.jsonl")) if probe_path.is_dir() else [probe_path]
    for candidate in paths:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    sample = json.loads(line)
                    break
        if sample is not None:
            break
    if sample is not None and ("before_frame" in sample or ("input" in sample and "target" in sample)):
        return ARCTransitionGNNDataset(path, config, progress_callback)
    return MockGNNDataset(path, config)


def batch_graph_examples(examples: list[tuple[GraphState, np.ndarray]]) -> tuple[GraphState, np.ndarray]:
    if not examples:
        raise ValueError("Cannot batch an empty list of graph examples.")
    if len(examples) == 1:
        return examples[0]

    nodes: list[ObjectComponent] = []
    node_features = []
    edge_attrs = []
    edge_indices = []
    graph_features = []
    labels = []
    node_offset = 0
    first_graph = examples[0][0]

    for graph, target in examples:
        nodes.extend(graph.nodes)
        node_features.append(np.asarray(graph.node_features, dtype=np.float32))
        labels.append(np.asarray(target, dtype=np.float32))
        edge_attr = np.asarray(graph.edge_attr, dtype=np.float32)
        edge_attrs.append(edge_attr)
        edge_index = np.asarray(graph.edge_index, dtype=np.int64)
        if edge_index.size:
            edge_indices.append(edge_index + node_offset)
        if graph.graph_features is not None:
            graph_features.append(np.asarray(graph.graph_features, dtype=np.float32))
        node_offset += graph.num_nodes

    edge_feature_dim = first_graph.edge_feature_dim
    batched_edge_attr = (
        np.concatenate(edge_attrs, axis=0)
        if edge_attrs
        else np.zeros((0, edge_feature_dim), dtype=np.float32)
    )
    batched_edge_index = (
        np.concatenate(edge_indices, axis=1)
        if edge_indices
        else np.zeros((2, 0), dtype=np.int64)
    )
    batched_graph_features = np.mean(np.stack(graph_features, axis=0), axis=0) if graph_features else None
    return (
        GraphState(
            nodes=nodes,
            node_features=np.concatenate(node_features, axis=0),
            edge_index=batched_edge_index,
            edge_attr=batched_edge_attr,
            graph_features=batched_graph_features,
            action_context=None,
            frame_index=first_graph.frame_index,
            feature_names=first_graph.feature_names,
            edge_feature_names=first_graph.edge_feature_names,
        ),
        np.concatenate(labels, axis=0),
    )


def _centroid_distance(a: ObjectComponent, b: ObjectComponent) -> float:
    return float(((a.centroid[0] - b.centroid[0]) ** 2 + (a.centroid[1] - b.centroid[1]) ** 2) ** 0.5)


def _index_jsonl_file(path: Path) -> list[tuple[Path, int]]:
    entries: list[tuple[Path, int]] = []
    with path.open("r", encoding="utf-8") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if line.strip():
                entries.append((path, offset))
    return entries
