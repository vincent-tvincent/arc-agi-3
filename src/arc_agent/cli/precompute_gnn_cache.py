#!/usr/bin/env python3
"""Precompute GNN graph/soft-label examples into disk cache shards."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.training.gnn_dataset import graph_cache_config_hash, graph_to_cache_dict, make_gnn_dataset
from arc_agent.utils.config import load_config
from arc_agent.utils.device import torch_available

_WORKER_DATASET: Any | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute GNN graph/label cache shards.")
    parser.add_argument("--config", default="configs/train_gnn_arc.yaml")
    parser.add_argument("--data-dir", default=None, help="Override source gnn_training.data_dir.")
    parser.add_argument("--data-format", default=None, choices=["auto", "mock", "arc", "official"])
    parser.add_argument("--out-dir", required=True, help="Output cache directory.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker processes.")
    parser.add_argument("--shard-size", type=int, default=2048, help="Examples per cache shard.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum examples for a smoke cache.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing cache directory.")
    args = parser.parse_args()

    if not torch_available():
        raise SystemExit("PyTorch is required because cache shards are written with torch.save.")
    import torch

    config = load_config(args.config)
    train_cfg = config.get("gnn_training", {})
    data_dir = Path(args.data_dir or train_cfg.get("data_dir", "runs/gnn_mock_data"))
    data_format = args.data_format or train_cfg.get("data_format", "auto")
    out_dir = Path(args.out_dir)
    workers = max(1, int(args.workers if args.workers is not None else config.get("hardware", {}).get("num_workers", 1)))
    shard_size = max(1, int(args.shard_size))

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Cache output already exists. Use --overwrite to replace it: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_config = _cache_worker_config(config)
    dataset = make_gnn_dataset(data_dir, worker_config, data_format, progress_callback=_event)
    total = len(dataset) if args.limit is None else min(len(dataset), int(args.limit))
    if total <= 0:
        raise SystemExit(f"No examples found in {data_dir}")

    manifest_entries: list[dict[str, Any]] = []
    shard: list[dict[str, Any]] = []
    shard_index = 0
    label_sums = [0.0 for _ in AFFORDANCE_NAMES]
    node_count = 0
    start = time.perf_counter()
    _event(
        f"precomputing GNN cache source={data_dir} format={data_format} rows={total} "
        f"workers={workers} shard_size={shard_size} out_dir={out_dir}"
    )

    indices = range(total)
    executor: ProcessPoolExecutor | None = None
    if workers == 1:
        _init_worker(str(data_dir), worker_config, data_format)
        iterator = map(_build_example, indices)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(data_dir), worker_config, data_format),
        )
        iterator = executor.map(_build_example, indices, chunksize=max(1, shard_size // max(workers * 4, 1)))

    try:
        for completed, example in enumerate(iterator, start=1):
            labels = example["labels"]
            for idx, value in enumerate(labels.sum(axis=0).tolist()):
                label_sums[idx] += float(value)
            node_count += int(labels.shape[0])
            shard.append(example)
            if len(shard) >= shard_size:
                shard_name = _write_shard(out_dir, shard_index, shard, torch)
                _append_manifest_entries(manifest_entries, shard_name, len(shard))
                shard_index += 1
                shard = []
            if completed == 1 or completed % 1000 == 0 or completed == total:
                elapsed = time.perf_counter() - start
                _event(f"precomputed rows={completed}/{total} rows_per_sec={completed / max(elapsed, 1e-9):.2f}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if shard:
        shard_name = _write_shard(out_dir, shard_index, shard, torch)
        _append_manifest_entries(manifest_entries, shard_name, len(shard))
        shard_index += 1

    manifest = {
        "format": "arc_agent_gnn_cache",
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
        "source_data_dir": str(data_dir),
        "source_data_format": data_format,
        "labeler_config_hash": graph_cache_config_hash(worker_config),
        "num_examples": total,
        "num_shards": shard_index,
        "shard_size": shard_size,
        "affordances": AFFORDANCE_NAMES,
        "label_statistics": {
            "node_count": node_count,
            "label_sums": label_sums,
        },
        "entries": manifest_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _event(f"wrote GNN cache manifest={out_dir / 'manifest.json'} shards={shard_index} rows={total}")


def _cache_worker_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = json.loads(json.dumps(config))
    train_cfg = cloned.setdefault("gnn_training", {})
    train_cfg["lazy_index"] = True
    train_cfg["index_workers"] = 1
    return cloned


def _init_worker(data_dir: str, config: dict[str, Any], data_format: str) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = make_gnn_dataset(data_dir, config, data_format)


def _build_example(index: int) -> dict[str, Any]:
    if _WORKER_DATASET is None:
        raise RuntimeError("cache worker dataset was not initialized")
    graph, labels = _WORKER_DATASET[index]
    return {
        "source_index": int(index),
        "graph": graph_to_cache_dict(graph),
        "labels": labels.astype("float32"),
    }


def _write_shard(out_dir: Path, shard_index: int, shard: list[dict[str, Any]], torch: Any) -> str:
    shard_name = f"shard_{shard_index:06d}.pt"
    torch.save(shard, out_dir / shard_name)
    return shard_name


def _append_manifest_entries(entries: list[dict[str, Any]], shard_name: str, count: int) -> None:
    entries.extend({"shard": shard_name, "index": index} for index in range(count))


def _event(message: str) -> None:
    print(f"[precompute_gnn_cache] {message}", flush=True)


if __name__ == "__main__":
    main()
