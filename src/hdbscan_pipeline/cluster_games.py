"""Cluster ARC-AGI-3 game runs from aggregated transition features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .vector_builder import FEATURE_COLUMNS, FeatureRecord, build_vector_from_file
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hdbscan_pipeline.vector_builder import FEATURE_COLUMNS, FeatureRecord, build_vector_from_file

SUMMARY_FEATURES = [
    "coordinate_action_rate",
    "frame_changed_rate",
    "changed_cells_mean",
    "small_change_rate_1_8",
    "large_change_rate_33_plus",
    "progress_rate",
    "game_over_rate",
    "win_rate",
    "unique_after_hash_rate",
    "initial_component_count",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "examples_dir": "training_examples",
    "out_dir": "clusters",
    "method": "auto",
    "min_cluster_size": 3,
    "min_samples": 2,
    "n_clusters": 5,
}


def main() -> None:
    args = parse_args()
    debug("config", format_config(args))

    example_paths = sorted(Path(args.examples_dir).glob("*.examples.jsonl"))
    if not example_paths:
        raise SystemExit(f"No training examples found in {args.examples_dir!r}.")
    debug("load", f"found {len(example_paths)} training example files in {args.examples_dir}")
    debug("load", f"first files: {format_path_preview(example_paths)}")

    analysis_ids, matrix, feature_names, records = load_feature_matrix_with_debug(example_paths)
    row_count = len(matrix)
    column_count = len(feature_names)
    transition_counts = [int(record.vector[feature_names.index("transition_count")]) for record in records]
    debug("vector", f"built matrix rows={row_count} columns={column_count}")
    debug("vector", f"analysis ids: {format_text_preview(analysis_ids)}")
    debug("vector", f"transition counts: {format_number_summary(transition_counts)}")
    debug("vector", f"feature preview: {', '.join(feature_names[:8])}")

    scaled = standardize_columns(matrix)
    constant_columns = count_constant_columns(matrix)
    debug("scale", f"standardized columns; constant columns={constant_columns}")

    debug(
        "cluster",
        (
            f"requested_method={args.method} min_cluster_size={args.min_cluster_size} "
            f"min_samples={args.min_samples} n_clusters={args.n_clusters}"
        ),
    )
    method, labels, probabilities = cluster_matrix(
        scaled,
        requested_method=args.method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters,
    )
    debug("cluster", f"actual_method={method}")
    debug("cluster", f"label distribution: {format_label_distribution(labels)}")
    debug("cluster", f"probabilities: {format_number_summary(probabilities)}")

    rows = build_assignment_rows(records, labels, probabilities)
    summaries = build_cluster_summaries(rows, records, labels)
    debug("cluster", format_cluster_preview(summaries))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug("write", f"writing outputs to {out_dir}")
    write_assignments_csv(out_dir / "game_clusters.csv", rows)
    write_summary_json(
        out_dir / "game_clusters.json",
        {
            "method": method,
            "examples_dir": args.examples_dir,
            "config": vars(args),
            "run_count": len(records),
            "feature_count": len(feature_names),
            "feature_columns": feature_names,
            "clusters": summaries,
        },
    )
    write_summary_json(out_dir / "feature_columns.json", {"feature_columns": list(FEATURE_COLUMNS)})

    print(f"method={method}")
    print(f"runs={len(records)} features={len(feature_names)}")
    print(f"clusters={sorted(set(labels))}")
    print(f"wrote {out_dir / 'game_clusters.csv'}")
    print(f"wrote {out_dir / 'game_clusters.json'}")
    print(f"wrote {out_dir / 'feature_columns.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster game/run feature vectors for unsupervised game-type discovery.")
    parser.add_argument("--config", help="YAML file containing clustering arguments.")
    parser.add_argument("--examples-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--method", choices=["auto", "hdbscan", "agglomerative"])
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--n-clusters", type=int, help="Used by agglomerative clustering.")

    raw_args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    if raw_args.config:
        config.update(load_yaml_config(Path(raw_args.config)))

    for key, value in vars(raw_args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value
    config["config"] = raw_args.config

    return argparse.Namespace(**config)


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file {path} must contain a YAML mapping.")

    unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown:
        raise SystemExit(f"Unknown config key(s) in {path}: {', '.join(unknown)}")
    return loaded


def load_feature_matrix_with_debug(
    paths: list[Path],
    progress_interval: int = 25,
) -> tuple[list[str], list[list[float]], list[str], list[FeatureRecord]]:
    records: list[FeatureRecord] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        records.append(build_vector_from_file(path))
        if index == 1 or index % progress_interval == 0 or index == total:
            debug("vector", f"processed {index}/{total} files; latest={path.name}")

    return (
        [record.analysis_id for record in records],
        [record.vector for record in records],
        list(FEATURE_COLUMNS),
        records,
    )


def debug(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def format_config(args: argparse.Namespace) -> str:
    parts = []
    for key in ["config", "examples_dir", "out_dir", "method", "min_cluster_size", "min_samples", "n_clusters"]:
        parts.append(f"{key}={getattr(args, key)!r}")
    return " ".join(parts)


def format_path_preview(paths: list[Path], limit: int = 3) -> str:
    shown = [str(path) for path in paths[:limit]]
    suffix = f", ... +{len(paths) - limit} more" if len(paths) > limit else ""
    return ", ".join(shown) + suffix


def format_text_preview(values: list[str], limit: int = 5) -> str:
    shown = values[:limit]
    suffix = f", ... +{len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def format_number_summary(values: list[int] | list[float]) -> str:
    if not values:
        return "empty"
    numeric = [float(value) for value in values]
    mean = sum(numeric) / len(numeric)
    return f"min={min(numeric):.3f} mean={mean:.3f} max={max(numeric):.3f}"


def count_constant_columns(matrix: list[list[float]]) -> int:
    if not matrix:
        return 0
    column_count = len(matrix[0])
    constant_count = 0
    for col in range(column_count):
        values = {row[col] for row in matrix}
        if len(values) <= 1:
            constant_count += 1
    return constant_count


def format_label_distribution(labels: list[int]) -> str:
    counts = Counter(labels)
    return ", ".join(f"{label}:{counts[label]}" for label in sorted(counts))


def format_cluster_preview(summaries: list[dict[str, Any]], limit: int = 8) -> str:
    if not summaries:
        return "no clusters to summarize"

    parts = []
    for summary in summaries[:limit]:
        cluster_id = summary["cluster_id"]
        size = summary["size"]
        members = summary["members"][:3]
        more = f", +{len(summary['members']) - 3} more" if len(summary["members"]) > 3 else ""
        parts.append(f"{cluster_id}: size={size} members={members}{more}")
    suffix = f"; ... +{len(summaries) - limit} clusters" if len(summaries) > limit else ""
    return "cluster preview: " + "; ".join(parts) + suffix


def cluster_matrix(
    matrix: list[list[float]],
    requested_method: str,
    min_cluster_size: int,
    min_samples: int,
    n_clusters: int,
) -> tuple[str, list[int], list[float]]:
    matrix_array = np.asarray(matrix, dtype=float)
    if requested_method in {"auto", "hdbscan"}:
        try:
            import hdbscan  # type: ignore[import-not-found]
        except ImportError:
            if requested_method == "hdbscan":
                raise SystemExit("hdbscan is not installed. Install it or use --method agglomerative.")
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = [int(label) for label in clusterer.fit_predict(matrix_array)]
            probabilities = [float(probability) for probability in clusterer.probabilities_]
            return "hdbscan", labels, probabilities

    labels = agglomerative_labels(matrix_array, n_clusters=n_clusters)
    return "agglomerative", labels, [1.0 for _ in labels]


def agglomerative_labels(matrix: np.ndarray[Any, np.dtype[np.float64]], n_clusters: int) -> list[int]:
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as error:
        raise SystemExit("scikit-learn is required for agglomerative clustering.") from error

    sample_count = matrix.shape[0]
    cluster_count = max(1, min(n_clusters, sample_count))
    model = AgglomerativeClustering(n_clusters=cluster_count)
    return [int(label) for label in model.fit_predict(matrix)]


def standardize_columns(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []

    row_count = len(matrix)
    column_count = len(matrix[0])
    means: list[float] = []
    stds: list[float] = []

    for col in range(column_count):
        values = [row[col] for row in matrix]
        mean = sum(values) / row_count
        variance = sum((value - mean) ** 2 for value in values) / row_count
        std = variance**0.5
        means.append(mean)
        stds.append(std if std > 0 else 1.0)

    return [
        [(value - means[col]) / stds[col] for col, value in enumerate(row)]
        for row in matrix
    ]


def build_assignment_rows(
    records: list[FeatureRecord],
    labels: list[int],
    probabilities: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, label, probability in zip(records, labels, probabilities, strict=True):
        features = dict(zip(record.feature_names, record.vector, strict=True))
        row = {
            "analysis_id": record.analysis_id,
            "game_id": record.game_id,
            "cluster_id": label,
            "cluster_probability": probability,
            "source_path": record.source_path,
        }
        for feature in SUMMARY_FEATURES:
            row[feature] = features.get(feature, 0.0)
        rows.append(row)
    return rows


def build_cluster_summaries(
    rows: list[dict[str, Any]],
    records: list[FeatureRecord],
    labels: list[int],
) -> list[dict[str, Any]]:
    records_by_id = {record.analysis_id: record for record in records}
    rows_by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_cluster[int(row["cluster_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for cluster_id in sorted(rows_by_cluster):
        cluster_rows = rows_by_cluster[cluster_id]
        member_ids = [str(row["analysis_id"]) for row in cluster_rows]
        member_records = [records_by_id[member_id] for member_id in member_ids]
        mean_features = mean_feature_values(member_records, SUMMARY_FEATURES)
        summaries.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster_rows),
                "members": member_ids,
                "mean_features": mean_features,
            }
        )
    return summaries


def mean_feature_values(records: list[FeatureRecord], feature_names: list[str]) -> dict[str, float]:
    if not records:
        return {feature: 0.0 for feature in feature_names}

    feature_maps = [dict(zip(record.feature_names, record.vector, strict=True)) for record in records]
    return {
        feature: sum(feature_map.get(feature, 0.0) for feature_map in feature_maps) / len(feature_maps)
        for feature in feature_names
    }


def write_assignments_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "analysis_id",
        "game_id",
        "cluster_id",
        "cluster_probability",
        "source_path",
        *SUMMARY_FEATURES,
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
