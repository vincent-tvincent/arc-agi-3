"""Graph datasets for affordance pretraining."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np

from arc_agent.envs.mock_grid_env import COLOR_LABELS
from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.perception.grid_extractor import frame_to_grid
from arc_agent.perception.object_tracker import ObjectTracker
from arc_agent.perception.object_types import ObjectComponent
from arc_agent.perception.segmenter import GridSegmenter

ARCIndexEntry: TypeAlias = tuple[Path, int] | tuple[Path, int, int]


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


class CachedGNNDataset:
    """Read precomputed graph/label shards created by precompute_gnn_cache."""

    def __init__(
        self,
        path: str | Path,
        config: dict[str, Any],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.config = config
        self.manifest_path = self.path / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing GNN cache manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries: list[dict[str, Any]] = list(self.manifest.get("entries", []))
        cache_cfg = config.get("gnn_training", {}).get("cache", {})
        self.max_open_shards = max(1, int(cache_cfg.get("max_open_shards", 4)))
        self._shard_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

        expected_hash = graph_cache_config_hash(config)
        actual_hash = str(self.manifest.get("labeler_config_hash", ""))
        if actual_hash and actual_hash != expected_hash and progress_callback is not None:
            progress_callback(
                "warning: GNN cache labeler_config_hash differs from current config "
                f"cache={actual_hash} current={expected_hash}; rebuild cache if labeler settings changed"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[GraphState, np.ndarray]:
        entry = self.entries[index]
        shard = self._load_shard(str(entry["shard"]))
        example = shard[int(entry["index"])]
        return graph_from_cache_dict(example["graph"]), np.asarray(example["labels"], dtype=np.float32)

    def split(self, validation_fraction: float) -> tuple[DatasetView, DatasetView]:
        indices = list(range(len(self)))
        train_cfg = self.config.get("gnn_training", {})
        if bool(train_cfg.get("split_shuffle", False)):
            rng = np.random.default_rng(int(self.config.get("project", {}).get("seed", 42)))
            rng.shuffle(indices)
        split_idx = int(len(self) * (1.0 - validation_fraction))
        return DatasetView(self, indices[:split_idx]), DatasetView(self, indices[split_idx:])

    def label_statistics(self) -> tuple[np.ndarray, int] | None:
        stats = self.manifest.get("label_statistics")
        if not isinstance(stats, dict):
            return None
        label_sums = stats.get("label_sums")
        node_count = stats.get("node_count")
        if label_sums is None or node_count is None:
            return None
        values = np.asarray(label_sums, dtype=np.float64)
        if values.shape[0] != len(AFFORDANCE_NAMES):
            return None
        return values, int(node_count)

    def _load_shard(self, shard_name: str) -> list[dict[str, Any]]:
        if shard_name in self._shard_cache:
            shard = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = shard
            return shard
        import torch

        path = self.path / shard_name
        try:
            shard = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            shard = torch.load(path, map_location="cpu")
        loaded_shard = list(shard)
        self._shard_cache[shard_name] = loaded_shard
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return loaded_shard


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
        self.entries: list[ARCIndexEntry] = []
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
        indices = list(range(len(self)))
        train_cfg = self.config.get("gnn_training", {})
        if bool(train_cfg.get("split_shuffle", False)):
            rng = np.random.default_rng(int(self.config.get("project", {}).get("seed", 42)))
            rng.shuffle(indices)
        split_idx = int(len(self) * (1.0 - validation_fraction))
        return DatasetView(self, indices[:split_idx]), DatasetView(self, indices[split_idx:])

    def _pseudo_labels(
        self,
        objects: list[ObjectComponent],
        before: np.ndarray,
        after: np.ndarray,
        row: dict[str, Any],
    ) -> np.ndarray:

        labels = np.zeros((len(objects), len(AFFORDANCE_NAMES)), dtype=np.float32)
        height, width = before.shape
        frame_area = max(height * width, 1)
        background_color = _background_color(before)

        label_cfg = self.config.get("gnn_training", {}).get("pseudo_labeling", {})

        changed_cells = {(int(cell["x"]), int(cell["y"])) for cell in row.get("changed_cells", [])}

        after_components = self.segmenter.segment(after, frame_index=int(row["step"]) + 1)
        after_by_color: dict[int, list[ObjectComponent]] = {}

        for component in after_components:
            if component.color_id is not None:
                after_by_color.setdefault(int(component.color_id), []).append(component)

        hud_container_bboxes = _hud_container_bboxes(objects, width, height, frame_area, label_cfg)

        for node_idx, node in enumerate(objects):
            object_cells = node.cells
            object_id = node.track_id or node.id
            del object_id
            intersects_change = bool(object_cells & changed_cells)
            adjacency_radius = int(label_cfg.get("direct_change_adjacency_radius", 1))
            adjacent_change = any(
                abs(x - cx) + abs(y - cy) <= adjacency_radius
                for x, y in object_cells
                for cx, cy in changed_cells
            )

            direct_change = intersects_change or adjacent_change

            floor_like = _is_floor_like(node, background_color, frame_area, label_cfg)
            status_like = _is_status_like(node, width, height, label_cfg)

            hud_like = _is_hud_like(node, width, height, label_cfg) or (
                bool(label_cfg.get("hud_container_inheritance", True))
                and _inside_any_bbox(node.bbox, hud_container_bboxes, int(label_cfg.get("hud_container_padding", 1)))
            )
            if status_like or (hud_like and bool(label_cfg.get("hud_like_is_status", True))):
                ui_key = "status_ui" if status_like else "hud_ui"
                if hud_like and not status_like:
                    ui_key = "hud_preview_ui" if _inside_any_bbox(node.bbox, hud_container_bboxes, int(label_cfg.get("hud_container_padding", 1))) else "hud_ui"
                _set_label(labels, node_idx, "ui_or_status", _soft_value(label_cfg, ui_key, 1.0))
                if hud_like:
                    _set_label(labels, node_idx, "movable", _soft_value(label_cfg, "hud_preview_movable_residual", 0.0))
                    _set_label(labels, node_idx, "unknown_interactable", _soft_value(label_cfg, "hud_preview_unknown_residual", 0.0))

            if floor_like or status_like or (hud_like and bool(label_cfg.get("hud_like_is_status", True))):
                continue

            if row.get("is_win") or int(row.get("level_delta", 0)) > 0:
                if direct_change:
                    _set_label(labels, node_idx, "goal_related", _soft_value(label_cfg, "win_goal_related", 1.0))

            if row.get("is_game_over"):
                if direct_change:
                    _set_label(labels, node_idx, "hazardous", _soft_value(label_cfg, "game_over_hazardous", 1.0))

            moved_similar = False
            if node.color_id is not None:
                after_same_color = after_by_color.get(int(node.color_id), [])
                similar_after = [component for component in after_same_color if _is_similar_component(node, component, label_cfg)]
                moved_similar_components = [
                    component
                    for component in similar_after
                    if component.centroid != node.centroid
                    and _centroid_distance(component, node) <= float(label_cfg.get("movable_max_centroid_distance", 8.0))
                    and _overlap_ratio(node, component) <= float(label_cfg.get("similar_max_motion_overlap_ratio", 0.85))
                ]

                if direct_change and not hud_like and not similar_after:
                    changed_to_background = any(
                        before[y, x] == node.color_id and after[y, x] == background_color
                        for x, y in object_cells & changed_cells
                        if 0 <= y < before.shape[0] and 0 <= x < before.shape[1]
                    )

                    if changed_to_background:
                        compactness = _compactness_score(node)
                        smallness = _smallness_score(node.area, int(label_cfg.get("collectible_max_area", 64)))
                        collectible_score = _soft_value(label_cfg, "collectible", 1.0) * _soft_modifier(
                            label_cfg, _mix_scores(compactness, smallness)
                        )
                        _set_label(labels, node_idx, "collectible", collectible_score)

                moved_similar = bool(moved_similar_components)

                if direct_change and not hud_like and moved_similar:
                    nearest_motion = min(_centroid_distance(component, node) for component in moved_similar_components)
                    controllable_step_distance = float(label_cfg.get("controllable_max_step_distance", 2.0))
                    if nearest_motion <= controllable_step_distance:
                        motion_score = _motion_step_score(
                            nearest_motion,
                            controllable_step_distance,
                            float(label_cfg.get("motion_score_floor", 0.5)),
                        )
                        motion_score = _soft_modifier(label_cfg, motion_score)
                        _set_label(labels, node_idx, "movable", _soft_value(label_cfg, "short_motion_movable", 1.0) * motion_score)
                        controllable_max_area = max(
                            int(label_cfg.get("controllable_max_area", 64)),
                            int(frame_area * float(label_cfg.get("controllable_max_area_ratio", 0.02))),
                        )

                        if node.area <= controllable_max_area:
                            smallness = _smallness_score(node.area, controllable_max_area)
                            controllable_score = (
                                _soft_value(label_cfg, "short_motion_controllable", 1.0)
                                * motion_score
                                * _soft_modifier(label_cfg, max(smallness, float(label_cfg.get("controllable_smallness_floor", 0.25))))
                            )
                            _set_label(labels, node_idx, "controllable", controllable_score)
                    elif _has_remote_change(node, changed_cells, label_cfg):
                        _set_label(labels, node_idx, "goal_related", _soft_value(label_cfg, "remote_target_goal_related", 1.0))
                        _set_label(labels, node_idx, "movable", _soft_value(label_cfg, "remote_target_movable_residual", 0.0))

            if (
                direct_change
                and not moved_similar
                and _has_remote_change(node, changed_cells, label_cfg)
                and _is_trigger_source_like(node, label_cfg)
            ):
                compactness = _compactness_score(node)
                smallness = _smallness_score(node.area, int(label_cfg.get("trigger_source_max_area", 32)))
                trigger_score = _soft_value(label_cfg, "trigger_source_trigger", 1.0) * _soft_modifier(
                    label_cfg, _mix_scores(compactness, smallness)
                )
                _set_label(labels, node_idx, "trigger", trigger_score)

            if not row.get("frame_changed", True) and adjacent_change:
                largeness = _largeness_score(node.area, int(frame_area * float(label_cfg.get("blocking_large_area_ratio", 0.04))))
                blocking_score = _soft_value(label_cfg, "blocking", 1.0) * _soft_modifier(
                    label_cfg, max(largeness, float(label_cfg.get("blocking_largeness_floor", 0.25)))
                )
                _set_label(labels, node_idx, "blocking", blocking_score)

            if direct_change and bool(label_cfg.get("unknown_interactable_as_fallback", True)):
                if labels[node_idx].sum() == 0.0:
                    _set_label(labels, node_idx, "unknown_interactable", _soft_value(label_cfg, "unknown_fallback", 1.0))
            elif direct_change:
                _set_label(labels, node_idx, "unknown_interactable", _soft_value(label_cfg, "unknown_fallback", 1.0))

        return labels

    def _index_rows(self, progress_callback: Callable[[str], None] | None = None) -> list[ARCIndexEntry]:
        paths = _data_paths(self.path)
        if self.index_workers <= 1 or len(paths) <= 1:
            return self._index_rows_serial(paths, progress_callback)

        entries_by_path: dict[Path, list[ARCIndexEntry]] = {}
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
        entries: list[ARCIndexEntry] = []
        for path in paths:
            entries.extend(entries_by_path.get(path, []))
        return entries

    def _index_rows_serial(
        self,
        paths: list[Path],
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[ARCIndexEntry]:
        entries: list[ARCIndexEntry] = []
        total_files = len(paths)
        for file_idx, path in enumerate(paths, start=1):
            entries.extend(_index_jsonl_file(path))
            if progress_callback is not None and (file_idx == 1 or file_idx % 100 == 0 or file_idx == total_files):
                progress_callback(f"indexed ARC jsonl files={file_idx}/{total_files} rows={len(entries)} workers=1")
        return entries

    def _read_entry(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        if len(entry) == 3:
            path, before_offset, after_offset = entry
            with path.open("r", encoding="utf-8") as file:
                file.seek(before_offset)
                before = json.loads(file.readline())
                file.seek(after_offset)
                after = json.loads(file.readline())
            return _gt_transition_from_records(before, after, index)
        path, offset = entry
        with path.open("r", encoding="utf-8") as file:
            file.seek(offset)
            return json.loads(file.readline())

    def _read_rows(self, progress_callback: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        paths = _data_paths(self.path)
        total_files = len(paths)
        for file_idx, path in enumerate(paths, start=1):
            rows.extend(_read_rows_from_file(path))
            if progress_callback is not None and (file_idx == 1 or file_idx % 100 == 0 or file_idx == total_files):
                progress_callback(f"loaded ARC jsonl files={file_idx}/{total_files} rows={len(rows)} lazy_index=false")
        return rows

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if "before_frame" in row and "after_frame" in row:
            return {
                "step": int(row.get("step", 0)),
                "before_frame": _frame_to_grid_list(row["before_frame"]),
                "after_frame": _frame_to_grid_list(row["after_frame"]),
                "changed_cells": row.get("changed_cells", []),
                "action": row.get("action"),
                "frame_changed": bool(row.get("frame_changed", row.get("before_hash") != row.get("after_hash"))),
                "is_win": bool(row.get("is_win", row.get("state") == "WIN")),
                "is_game_over": bool(row.get("is_game_over", row.get("state") == "GAME_OVER")),
                "level_delta": int(row.get("level_delta", 0)),
            }
        if "input" in row and "target" in row:
            outcome = row.get("outcome", {})
            return {
                "step": int(row.get("step", 0)),
                "before_frame": _frame_to_grid_list(row["input"].get("frame")),
                "after_frame": _frame_to_grid_list(row["target"].get("next_frame")),
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
    if data_format == "cache":
        return CachedGNNDataset(path, config, progress_callback)
    if data_format in {"arc", "arc_transitions", "official"}:
        return ARCTransitionGNNDataset(path, config, progress_callback)
    probe_path = Path(path)
    if probe_path.is_dir() and (probe_path / "manifest.json").exists():
        return CachedGNNDataset(path, config, progress_callback)
    sample: dict[str, Any] | None = None
    paths = _data_paths(probe_path)
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
    if sample is not None and (
        "before_frame" in sample
        or ("input" in sample and "target" in sample)
        or _is_gt_record(sample)
    ):
        return ARCTransitionGNNDataset(path, config, progress_callback)
    return MockGNNDataset(path, config)


def graph_cache_config_hash(config: dict[str, Any]) -> str:
    """Hash config sections that affect graph construction and pseudo-labels."""

    relevant = {
        "perception": config.get("perception", {}),
        "tracking": config.get("tracking", {}),
        "features": config.get("features", {}),
        "edge_builder": config.get("edge_builder", {}),
        "model_affordances": config.get("model", {}).get("affordances", AFFORDANCE_NAMES),
        "pseudo_labeling": config.get("gnn_training", {}).get("pseudo_labeling", {}),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def graph_to_cache_dict(graph: GraphState) -> dict[str, Any]:
    return {
        "nodes": [_object_to_cache_dict(node) for node in graph.nodes],
        "node_features": np.asarray(graph.node_features, dtype=np.float32),
        "edge_index": np.asarray(graph.edge_index, dtype=np.int64),
        "edge_attr": np.asarray(graph.edge_attr, dtype=np.float32),
        "graph_features": None if graph.graph_features is None else np.asarray(graph.graph_features, dtype=np.float32),
        "action_context": None if graph.action_context is None else np.asarray(graph.action_context, dtype=np.float32),
        "frame_index": int(graph.frame_index),
        "feature_names": list(graph.feature_names),
        "edge_feature_names": list(graph.edge_feature_names),
    }


def graph_from_cache_dict(data: dict[str, Any]) -> GraphState:
    return GraphState(
        nodes=[_object_from_cache_dict(item) for item in data.get("nodes", [])],
        node_features=np.asarray(data["node_features"], dtype=np.float32),
        edge_index=np.asarray(data["edge_index"], dtype=np.int64),
        edge_attr=np.asarray(data["edge_attr"], dtype=np.float32),
        graph_features=None if data.get("graph_features") is None else np.asarray(data["graph_features"], dtype=np.float32),
        action_context=None if data.get("action_context") is None else np.asarray(data["action_context"], dtype=np.float32),
        frame_index=int(data.get("frame_index", 0)),
        feature_names=list(data.get("feature_names", [])),
        edge_feature_names=list(data.get("edge_feature_names", [])),
    )


def _object_to_cache_dict(node: ObjectComponent) -> dict[str, Any]:
    return {
        "id": int(node.id),
        "color_id": None if node.color_id is None else int(node.color_id),
        "pixels": np.asarray(node.pixels, dtype=np.int64),
        "bbox": tuple(int(value) for value in node.bbox),
        "centroid": tuple(float(value) for value in node.centroid),
        "area": int(node.area),
        "frame_index": int(node.frame_index),
        "track_id": None if node.track_id is None else int(node.track_id),
        "attributes": dict(node.attributes),
    }


def _object_from_cache_dict(data: dict[str, Any]) -> ObjectComponent:
    return ObjectComponent(
        id=int(data["id"]),
        color_id=None if data.get("color_id") is None else int(data["color_id"]),
        pixels=np.asarray(data["pixels"], dtype=np.int64),
        bbox=tuple(int(value) for value in data["bbox"]),
        centroid=tuple(float(value) for value in data["centroid"]),
        area=int(data["area"]),
        frame_index=int(data.get("frame_index", 0)),
        track_id=None if data.get("track_id") is None else int(data["track_id"]),
        attributes=dict(data.get("attributes", {})),
    )


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


def _overlap_ratio(a: ObjectComponent, b: ObjectComponent) -> float:
    return len(a.cells & b.cells) / max(min(a.area, b.area), 1)


def _set_label(labels: np.ndarray, node_idx: int, name: str, confidence: float) -> None:
    labels[node_idx, AFFORDANCE_NAMES.index(name)] = max(
        float(labels[node_idx, AFFORDANCE_NAMES.index(name)]),
        _clamp01(confidence),
    )


def _soft_value(config: dict[str, Any], key: str, hard_value: float) -> float:
    if not _soft_enabled(config):
        return hard_value
    soft_cfg = config.get("soft_labels", {})
    return _clamp01(float(soft_cfg.get(key, hard_value)))


def _soft_modifier(config: dict[str, Any], value: float) -> float:
    return _clamp01(value) if _soft_enabled(config) else 1.0


def _soft_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("soft_labels", {}).get("enabled", False))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _smallness_score(area: int, max_area: int) -> float:
    return 1.0 - _clamp01(area / max(max_area, 1))


def _largeness_score(area: int, large_area: int) -> float:
    return _clamp01(area / max(large_area, 1))


def _compactness_score(node: ObjectComponent) -> float:
    return _clamp01(min(node.width, node.height) / max(node.width, node.height, 1))


def _motion_step_score(distance: float, max_distance: float, floor: float) -> float:
    scaled = 1.0 - _clamp01(distance / max(max_distance, 1e-6))
    return _clamp01(max(floor, scaled))


def _mix_scores(*scores: float) -> float:
    if not scores:
        return 1.0
    return _clamp01(sum(_clamp01(score) for score in scores) / len(scores))


def _background_color(frame: np.ndarray) -> int:
    values, counts = np.unique(frame, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _is_floor_like(node: ObjectComponent, background_color: int, frame_area: int, config: dict[str, Any]) -> bool:
    area_ratio = float(node.area / max(frame_area, 1))
    return node.color_id == background_color or area_ratio >= float(config.get("floor_area_ratio", 0.18))


def _is_status_like(node: ObjectComponent, width: int, height: int, config: dict[str, Any]) -> bool:
    node_width = max(node.width, 1)
    node_height = max(node.height, 1)
    edge_margin = int(config.get("status_edge_margin", 1))
    min_long_abs = int(config.get("status_min_long_abs", 8))
    max_thin_abs = int(config.get("status_max_thin_abs", 4))
    max_thin_ratio = float(config.get("status_max_thin_ratio", 0.07))
    near_edge = (
        node.bbox[0] <= edge_margin
        or node.bbox[1] <= edge_margin
        or node.bbox[2] >= width - 1 - edge_margin
        or node.bbox[3] >= height - 1 - edge_margin
    )
    long_horizontal = node_width >= max(min_long_abs, int(width * float(config.get("status_long_horizontal_ratio", 0.35)))) and node_height <= max(max_thin_abs, int(height * max_thin_ratio))
    long_vertical = node_height >= max(min_long_abs, int(height * float(config.get("status_long_vertical_ratio", 0.35)))) and node_width <= max(max_thin_abs, int(width * max_thin_ratio))
    bottom_progress = (
        node.bbox[1] >= int(height * float(config.get("status_bottom_band_ratio", 0.90)))
        and node_width >= max(min_long_abs, int(width * float(config.get("status_bottom_min_width_ratio", 0.25))))
        and node_height <= max(max_thin_abs, int(height * max_thin_ratio))
    )
    return bottom_progress or (near_edge and (long_horizontal or long_vertical))


def _is_hud_like(node: ObjectComponent, width: int, height: int, config: dict[str, Any]) -> bool:
    bottom_band = node.bbox[1] >= int(height * float(config.get("hud_bottom_band_ratio", 0.78)))
    bottom_left = (
        node.bbox[2] <= int(width * float(config.get("hud_bottom_left_width_ratio", 0.22)))
        and node.bbox[1] >= int(height * float(config.get("hud_bottom_left_y_ratio", 0.70)))
    )
    return bottom_band or bottom_left


def _hud_container_bboxes(
    nodes: list[ObjectComponent],
    width: int,
    height: int,
    frame_area: int,
    config: dict[str, Any],
) -> list[tuple[int, int, int, int]]:
    return [
        node.bbox
        for node in nodes
        if _is_hud_like(node, width, height, config) or _is_hud_preview_container_like(node, width, height, frame_area, config)
    ]


def _is_hud_preview_container_like(
    node: ObjectComponent,
    width: int,
    height: int,
    frame_area: int,
    config: dict[str, Any],
) -> bool:
    min_size = int(config.get("hud_preview_min_size", 6))
    max_area = int(frame_area * float(config.get("hud_preview_max_area_ratio", 0.08)))
    position_ok = True
    if bool(config.get("hud_preview_use_position_filter", True)):
        x_margin = int(width * float(config.get("hud_preview_edge_margin_ratio", 0.20)))
        y_margin = int(height * float(config.get("hud_preview_edge_margin_ratio", 0.20)))
        position_ok = (
            node.bbox[0] <= x_margin
            or node.bbox[1] <= y_margin
            or node.bbox[2] >= width - 1 - x_margin
            or node.bbox[3] >= height - 1 - y_margin
        )
    return (
        position_ok
        and node.width >= min_size
        and node.height >= min_size
        and node.area <= max_area
    )


def _inside_any_bbox(
    bbox: tuple[int, int, int, int],
    containers: list[tuple[int, int, int, int]],
    padding: int,
) -> bool:
    x0, y0, x1, y1 = bbox
    for cx0, cy0, cx1, cy1 in containers:
        if bbox == (cx0, cy0, cx1, cy1):
            continue
        if x0 >= cx0 - padding and y0 >= cy0 - padding and x1 <= cx1 + padding and y1 <= cy1 + padding:
            return True
    return False


def _is_trigger_source_like(node: ObjectComponent, config: dict[str, Any]) -> bool:
    max_area = int(config.get("trigger_source_max_area", 32))
    max_width = int(config.get("trigger_source_max_width", 8))
    max_height = int(config.get("trigger_source_max_height", 8))
    return node.area <= max_area and node.width <= max_width and node.height <= max_height


def _is_similar_component(before: ObjectComponent, after: ObjectComponent, config: dict[str, Any]) -> bool:
    area_ratio = after.area / max(before.area, 1)
    width_delta = abs(after.width - before.width)
    height_delta = abs(after.height - before.height)
    return (
        float(config.get("similar_min_area_ratio", 0.5)) <= area_ratio <= float(config.get("similar_max_area_ratio", 2.0))
        and width_delta <= max(int(config.get("similar_max_shape_delta_abs", 2)), before.width)
        and height_delta <= max(int(config.get("similar_max_shape_delta_abs", 2)), before.height)
    )


def _has_remote_change(node: ObjectComponent, changed_cells: set[tuple[int, int]], config: dict[str, Any]) -> bool:
    if not changed_cells:
        return False
    adjacency_radius = int(config.get("direct_change_adjacency_radius", 1))
    local_change_count = sum(
        1
        for cx, cy in changed_cells
        if any(abs(x - cx) + abs(y - cy) <= adjacency_radius for x, y in node.cells)
    )
    return len(changed_cells) >= int(config.get("trigger_min_changed_cells", 2)) and local_change_count > 0 and local_change_count < len(changed_cells)


def _data_paths(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() in {".jsonl", ".json"})


def _index_jsonl_file(path: Path) -> list[ARCIndexEntry]:
    first_row = _first_json_row(path)
    if first_row is not None and _is_gt_record(first_row):
        return list(_index_gt_file(path))
    entries: list[ARCIndexEntry] = []
    with path.open("r", encoding="utf-8") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if line.strip():
                entries.append((path, offset))
    return entries


def _index_gt_file(path: Path) -> list[ARCIndexEntry]:
    entries: list[ARCIndexEntry] = []
    previous_offset: int | None = None
    previous_row: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if previous_offset is not None and previous_row is not None and _same_gt_episode(previous_row, row):
                entries.append((path, previous_offset, offset))
            previous_offset = offset
            previous_row = row
    return entries


def _read_rows_from_file(path: Path) -> list[dict[str, Any]]:
    first_row = _first_json_row(path)
    if first_row is not None and _is_gt_record(first_row):
        return _read_gt_transitions(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_gt_transitions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for record in _iter_json_rows(path):
        if previous is not None and _same_gt_episode(previous, record):
            rows.append(_gt_transition_from_records(previous, record, len(rows)))
        previous = record
    return rows


def _iter_json_rows(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _first_json_row(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)
    return None


def _is_gt_record(row: dict[str, Any]) -> bool:
    data = row.get("data")
    return isinstance(data, dict) and "frame" in data and "action_input" in data


def _same_gt_episode(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_data = before.get("data", {})
    after_data = after.get("data", {})
    return (
        before_data.get("guid") == after_data.get("guid")
        and before_data.get("game_id") == after_data.get("game_id")
        and _has_valid_gt_frame(before)
        and _has_valid_gt_frame(after)
    )


def _gt_transition_from_records(before_record: dict[str, Any], after_record: dict[str, Any], step: int) -> dict[str, Any]:
    before_data = before_record["data"]
    after_data = after_record["data"]
    before_frame = _frame_to_grid_array(before_data.get("frame"))
    after_frame = _frame_to_grid_array(after_data.get("frame"))
    level_before = int(before_data.get("levels_completed", 0))
    level_after = int(after_data.get("levels_completed", 0))
    return {
        "step": step,
        "before_frame": before_frame.tolist(),
        "after_frame": after_frame.tolist(),
        "changed_cells": _changed_cells(before_frame, after_frame),
        "action": after_data.get("action_input", {}).get("id"),
        "frame_changed": not np.array_equal(before_frame, after_frame),
        "is_win": after_data.get("state") == "WIN",
        "is_game_over": after_data.get("state") == "GAME_OVER",
        "level_delta": level_after - level_before,
    }


def _frame_to_grid_array(frame: Any) -> np.ndarray:
    return np.asarray(frame_to_grid(frame), dtype=np.int64)


def _frame_to_grid_list(frame: Any) -> list[list[int]]:
    return _frame_to_grid_array(frame).tolist()


def _changed_cells(before: np.ndarray, after: np.ndarray) -> list[dict[str, int]]:
    if before.shape != after.shape:
        raise ValueError(f"Cannot diff frames with different shapes: {before.shape} != {after.shape}")
    changed: list[dict[str, int]] = []
    ys, xs = np.nonzero(before != after)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        changed.append({"x": int(x), "y": int(y), "before": int(before[y, x]), "after": int(after[y, x])})
    return changed


def _has_valid_gt_frame(record: dict[str, Any]) -> bool:
    try:
        frame = _frame_to_grid_array(record.get("data", {}).get("frame"))
    except Exception:
        return False
    return frame.ndim == 2 and frame.size > 0
