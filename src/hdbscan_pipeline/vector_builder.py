"""Build per-run feature vectors from step-level training examples.

The HDBSCAN stage should cluster one vector per game/run, not one vector per
transition. This module aggregates a full ``*.examples.jsonl`` file into a
stable numeric vector that later clustering code can load directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

try:
    from arc3_pipeline.frame_utils import background_color, color_counts, connected_components
except ModuleNotFoundError:
    from src.arc3_pipeline.frame_utils import background_color, color_counts, connected_components

ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]

BASE_FEATURE_COLUMNS = [
    "transition_count",
    "available_action_count_mean",
    "available_action_count_min",
    "available_action_count_max",
    "coordinate_action_rate",
    "coordinate_action_unique_xy_rate",
    "coordinate_action_mean_x",
    "coordinate_action_mean_y",
    "coordinate_action_std_x",
    "coordinate_action_std_y",
    "frame_changed_rate",
    "no_change_rate",
    "changed_cells_mean",
    "changed_cells_median",
    "changed_cells_std",
    "changed_cells_min",
    "changed_cells_max",
    "small_change_rate_1_8",
    "medium_change_rate_9_32",
    "large_change_rate_33_plus",
    "unique_before_hash_rate",
    "unique_after_hash_rate",
    "repeat_after_hash_rate",
    "progress_rate",
    "game_over_rate",
    "win_rate",
    "mean_level_delta",
    "max_level_delta",
    "initial_color_count",
    "initial_background_color",
    "initial_component_count",
    "initial_component_size_mean",
    "initial_component_size_median",
    "initial_component_size_std",
    "initial_component_size_min",
    "initial_component_size_max",
    "initial_largest_component_fraction",
]

PER_ACTION_FEATURE_SUFFIXES = [
    "used_rate",
    "frame_changed_rate",
    "changed_cells_mean",
    "progress_rate",
    "game_over_rate",
    "win_rate",
]

ACTION6_COORDINATE_FEATURE_COLUMNS = [
    "ACTION6_coordinate_rate",
    "ACTION6_unique_xy_rate",
    "ACTION6_mean_x",
    "ACTION6_mean_y",
    "ACTION6_std_x",
    "ACTION6_std_y",
]

FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + [f"{action}_{suffix}" for action in ACTIONS for suffix in PER_ACTION_FEATURE_SUFFIXES]
    + ACTION6_COORDINATE_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class FeatureRecord:
    """One clustering-ready feature vector for one game/run."""

    source_path: str
    analysis_id: str
    game_id: str
    feature_names: list[str]
    vector: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "analysis_id": self.analysis_id,
            "game_id": self.game_id,
            "features": dict(zip(self.feature_names, self.vector, strict=True)),
        }


def load_training_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_vector_from_file(path: str | Path) -> FeatureRecord:
    source = Path(path)
    rows = load_training_rows(source)
    vector_by_name = build_vector_from_rows(rows)
    first = rows[0] if rows else {}
    return FeatureRecord(
        source_path=str(source),
        analysis_id=str(first.get("analysis_id", source.stem.replace(".examples", ""))),
        game_id=str(first.get("game_id", "")),
        feature_names=list(FEATURE_COLUMNS),
        vector=[vector_by_name[name] for name in FEATURE_COLUMNS],
    )


def build_vector_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    vector: dict[str, float] = {}

    available_counts = [len(row["input"]["available_actions"]) for row in rows]
    changed_counts = [int(row["outcome"]["changed_cell_count"]) for row in rows]
    level_deltas = [int(row["outcome"]["level_delta"]) for row in rows]

    coordinate_rows = [row for row in rows if _has_xy(row)]
    coordinate_pairs = [_xy_pair(row) for row in coordinate_rows]
    coordinate_xs = [x for x, _ in coordinate_pairs]
    coordinate_ys = [y for _, y in coordinate_pairs]

    vector["transition_count"] = float(n)

    available_stats = _stats(available_counts)
    vector["available_action_count_mean"] = available_stats["mean"]
    vector["available_action_count_min"] = available_stats["min"]
    vector["available_action_count_max"] = available_stats["max"]

    vector["coordinate_action_rate"] = _rate(len(coordinate_rows), n)
    vector["coordinate_action_unique_xy_rate"] = _rate(len(set(coordinate_pairs)), len(coordinate_rows))
    vector["coordinate_action_mean_x"] = _stats(coordinate_xs)["mean"]
    vector["coordinate_action_mean_y"] = _stats(coordinate_ys)["mean"]
    vector["coordinate_action_std_x"] = _stats(coordinate_xs)["std"]
    vector["coordinate_action_std_y"] = _stats(coordinate_ys)["std"]

    changed_count_stats = _stats(changed_counts)
    vector["frame_changed_rate"] = _rate(sum(_frame_changed(row) for row in rows), n)
    vector["no_change_rate"] = 1.0 - vector["frame_changed_rate"] if n else 0.0
    vector["changed_cells_mean"] = changed_count_stats["mean"]
    vector["changed_cells_median"] = changed_count_stats["median"]
    vector["changed_cells_std"] = changed_count_stats["std"]
    vector["changed_cells_min"] = changed_count_stats["min"]
    vector["changed_cells_max"] = changed_count_stats["max"]
    vector["small_change_rate_1_8"] = _rate(sum(1 <= count <= 8 for count in changed_counts), n)
    vector["medium_change_rate_9_32"] = _rate(sum(9 <= count <= 32 for count in changed_counts), n)
    vector["large_change_rate_33_plus"] = _rate(sum(count >= 33 for count in changed_counts), n)

    before_hashes = [row["metadata"]["before_hash"] for row in rows]
    after_hashes = [row["metadata"]["after_hash"] for row in rows]
    vector["unique_before_hash_rate"] = _rate(len(set(before_hashes)), n)
    vector["unique_after_hash_rate"] = _rate(len(set(after_hashes)), n)
    vector["repeat_after_hash_rate"] = 1.0 - vector["unique_after_hash_rate"] if n else 0.0

    vector["progress_rate"] = _rate(sum(delta > 0 for delta in level_deltas), n)
    vector["game_over_rate"] = _rate(sum(_is_game_over(row) for row in rows), n)
    vector["win_rate"] = _rate(sum(_is_win(row) for row in rows), n)
    vector["mean_level_delta"] = _stats(level_deltas)["mean"]
    vector["max_level_delta"] = _stats(level_deltas)["max"]

    vector.update(_initial_frame_features(rows[0]["input"]["frame"] if rows else []))
    vector.update(_per_action_features(rows))

    return {name: float(vector.get(name, 0.0)) for name in FEATURE_COLUMNS}


def load_feature_records(paths: Iterable[str | Path]) -> list[FeatureRecord]:
    return [build_vector_from_file(path) for path in sorted(Path(path) for path in paths)]


def load_feature_matrix(paths: Iterable[str | Path]) -> tuple[list[str], list[list[float]], list[str], list[FeatureRecord]]:
    """Return ``analysis_ids, matrix, feature_names, records`` for clustering."""

    records = load_feature_records(paths)
    return (
        [record.analysis_id for record in records],
        [record.vector for record in records],
        list(FEATURE_COLUMNS),
        records,
    )


def _initial_frame_features(frame: list[list[int]]) -> dict[str, float]:
    counts = color_counts(frame)
    bg = background_color(frame)
    components = connected_components(frame, ignore_color=bg)
    component_sizes = [int(component["size"]) for component in components]
    component_size_stats = _stats(component_sizes)
    total_cells = sum(len(row) for row in frame)
    largest_component = max(component_sizes) if component_sizes else 0

    return {
        "initial_color_count": float(len(counts)),
        "initial_background_color": float(bg if bg is not None else -1),
        "initial_component_count": float(len(components)),
        "initial_component_size_mean": component_size_stats["mean"],
        "initial_component_size_median": component_size_stats["median"],
        "initial_component_size_std": component_size_stats["std"],
        "initial_component_size_min": component_size_stats["min"],
        "initial_component_size_max": component_size_stats["max"],
        "initial_largest_component_fraction": _rate(largest_component, total_cells),
    }


def _per_action_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    vector: dict[str, float] = {}

    for action in ACTIONS:
        action_rows = [row for row in rows if row["action"]["name"] == action]
        action_changed_counts = [int(row["outcome"]["changed_cell_count"]) for row in action_rows]
        action_level_deltas = [int(row["outcome"]["level_delta"]) for row in action_rows]

        vector[f"{action}_used_rate"] = _rate(len(action_rows), n)
        vector[f"{action}_frame_changed_rate"] = _rate(sum(_frame_changed(row) for row in action_rows), len(action_rows))
        vector[f"{action}_changed_cells_mean"] = _stats(action_changed_counts)["mean"]
        vector[f"{action}_progress_rate"] = _rate(sum(delta > 0 for delta in action_level_deltas), len(action_rows))
        vector[f"{action}_game_over_rate"] = _rate(sum(_is_game_over(row) for row in action_rows), len(action_rows))
        vector[f"{action}_win_rate"] = _rate(sum(_is_win(row) for row in action_rows), len(action_rows))

        if action == "ACTION6":
            coordinate_rows = [row for row in action_rows if _has_xy(row)]
            coordinate_pairs = [_xy_pair(row) for row in coordinate_rows]
            xs = [x for x, _ in coordinate_pairs]
            ys = [y for _, y in coordinate_pairs]
            vector["ACTION6_coordinate_rate"] = _rate(len(coordinate_rows), len(action_rows))
            vector["ACTION6_unique_xy_rate"] = _rate(len(set(coordinate_pairs)), len(coordinate_rows))
            vector["ACTION6_mean_x"] = _stats(xs)["mean"]
            vector["ACTION6_mean_y"] = _stats(ys)["mean"]
            vector["ACTION6_std_x"] = _stats(xs)["std"]
            vector["ACTION6_std_y"] = _stats(ys)["std"]

    return vector


def _stats(values: Iterable[int | float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(numbers)),
        "median": float(median(numbers)),
        "std": float(pstdev(numbers)) if len(numbers) > 1 else 0.0,
        "min": float(min(numbers)),
        "max": float(max(numbers)),
    }


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _has_xy(row: dict[str, Any]) -> bool:
    data = row["action"]["data"]
    return "x" in data and "y" in data


def _xy_pair(row: dict[str, Any]) -> tuple[int, int]:
    data = row["action"]["data"]
    return int(data["x"]), int(data["y"])


def _frame_changed(row: dict[str, Any]) -> bool:
    return bool(row["outcome"]["frame_changed"])


def _is_game_over(row: dict[str, Any]) -> bool:
    return bool(row["outcome"]["is_game_over"])


def _is_win(row: dict[str, Any]) -> bool:
    return bool(row["outcome"]["is_win"])
