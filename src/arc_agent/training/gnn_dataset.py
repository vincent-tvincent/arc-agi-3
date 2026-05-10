"""Synthetic/mock graph dataset for affordance pretraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from arc_agent.envs.mock_grid_env import COLOR_LABELS
from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.perception.object_tracker import ObjectTracker
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
    def __init__(self, dataset: MockGNNDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, np.ndarray]:
        return self.dataset[self.indices[index]]
