#!/usr/bin/env python3
"""Train the GNN affordance model on mock labels or official ARC rollouts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.models.gnn_affordance import GNNAffordanceModel
from arc_agent.training.checkpointing import CheckpointManager, build_checkpoint
from arc_agent.training.gnn_dataset import batch_graph_examples, make_gnn_dataset
from arc_agent.training.logging_utils import RunLogger
from arc_agent.training.metrics import binary_f1
from arc_agent.utils.config import load_config, run_dir
from arc_agent.utils.device import select_device, torch_available
from arc_agent.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_gnn.yaml")
    parser.add_argument("--data-dir", default=None, help="Override gnn_training.data_dir.")
    parser.add_argument("--data-format", default=None, choices=["auto", "mock", "arc", "official", "cache"], help="Override gnn_training.data_format.")
    parser.add_argument("--resume", default=None, help="Resume a GNN checkpoint, including optimizer state and epoch.")
    parser.add_argument(
        "--pretrained-gnn",
        default=None,
        help="Warm-start model weights from a GNN checkpoint while resetting optimizer state and epoch.",
    )
    args = parser.parse_args()
    if args.resume and args.pretrained_gnn:
        raise SystemExit("Use either --resume or --pretrained-gnn, not both.")
    if not torch_available():
        raise SystemExit("PyTorch is required for GNN training. Install with `pip install -e .[train]`.")
    import torch

    config = load_config(args.config)
    set_global_seed(int(config.get("project", {}).get("seed", 42)))
    device = select_device(config.get("hardware", {}).get("device", "auto"))
    root = run_dir(config)
    logger = RunLogger(config.get("project", {}).get("run_name", "train_gnn"), root / "logs", True, config.get("logging", {}).get("use_tensorboard", False))
    manager = CheckpointManager(root / "checkpoints", int(config.get("storage", {}).get("checkpoint_keep_last", 3)))
    train_cfg = config.get("gnn_training", {})
    data_dir = args.data_dir or train_cfg.get("data_dir", "/run/media/blue-lobster/disk3/CS274p_output/runs/gnn_mock_data")
    data_format = args.data_format or train_cfg.get("data_format", "auto")
    logger.event(f"starting GNN setup config={args.config} data_dir={data_dir} data_format={data_format} requested_device={config.get('hardware', {}).get('device', 'auto')} selected_device={device}")
    logger.event("indexing/loading GNN dataset; official ARC JSONL indexing is CPU/disk work and happens before GPU training starts")
    dataset = make_gnn_dataset(data_dir, config, data_format, progress_callback=logger.event)
    if len(dataset) == 0:
        raise SystemExit(
            "No GNN training data found. For mock data run "
            "`python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000`; "
            "for official ARC data run "
            "`python src/experience_collection/collect_experience.py --game all --steps 200 --offline "
            "--out-dir /run/media/blue-lobster/disk3/CS274p_output/training_runs "
            "--training-out-dir /run/media/blue-lobster/disk3/CS274p_output/training_examples`."
        )
    train_set, val_set = dataset.split(float(config.get("gnn_training", {}).get("validation_fraction", 0.2)))
    logger.event(
        f"loaded dataset type={dataset.__class__.__name__} data_dir={data_dir} "
        f"data_format={data_format} rows={len(dataset)} train={len(train_set)} val={len(val_set)} device={device}"
    )
    graph0, _ = dataset[0]
    model_cfg = config.get("model", {}).get("gnn", {})

    model = GNNAffordanceModel(
        node_in_dim=graph0.node_feature_dim,
        edge_in_dim=graph0.edge_feature_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        layers=int(model_cfg.get("layers", 3)),
        num_affordances=len(AFFORDANCE_NAMES),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("gnn_training", {}).get("learning_rate", 0.001)))
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        checkpoint = manager.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch") or 0) + 1
        best_f1 = float(checkpoint.get("metrics", {}).get("val/mean_f1", 0.0))
        logger.event(f"resumed GNN checkpoint from {args.resume} start_epoch={start_epoch}")
    elif args.pretrained_gnn:
        checkpoint = manager.load(args.pretrained_gnn, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.event(f"loaded pretrained GNN weights from {args.pretrained_gnn}; optimizer reset and training starts at epoch 0")

    epochs = int(config.get("gnn_training", {}).get("epochs", 10))
    batch_size = max(1, int(config.get("gnn_training", {}).get("batch_size", 1)))
    shuffle = bool(config.get("gnn_training", {}).get("shuffle", False))
    progress_interval = max(1, int(config.get("logging", {}).get("log_interval_steps", 1000)))
    pos_weight = build_positive_weight(train_set, config.get("gnn_training", {}), device, logger)
    loader_workers = _dataloader_workers(config)
    logger.event(
        "GNN dataloader "
        f"enabled={bool(config.get('gnn_training', {}).get('use_dataloader', True))} "
        f"workers={loader_workers} batch_size={batch_size} shuffle={shuffle}"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        losses: list[float] = []
        epoch_start = time.perf_counter()
        completed = 0
        train_loader = make_graph_loader(train_set, config, batch_size, shuffle=shuffle, epoch=epoch)
        for graph, labels, example_count in train_loader:
            torch_graph = graph.to_torch(device)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            output = model(torch_graph)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output.affordance_logits,
                target,
                pos_weight=pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            completed += int(example_count)
            if len(losses) == 1 or completed % progress_interval == 0 or completed == len(train_set):
                elapsed = time.perf_counter() - epoch_start
                rows_per_sec = completed / max(elapsed, 1e-9)
                logger.log_metrics(
                    "train_gnn_progress",
                    epoch * max(len(train_set), 1) + completed,
                    {
                        "epoch": epoch,
                        "row": completed,
                        "rows": len(train_set),
                        "batch_size": batch_size,
                        "nodes": graph.num_nodes,
                        "loss": round(float(np.mean(losses[-min(len(losses), progress_interval) :])), 6),
                        "rows_per_sec": round(rows_per_sec, 2),
                    },
                )
        label_threshold = float(
            config.get("gnn_training", {}).get("pseudo_labeling", {}).get("soft_labels", {}).get("metric_label_threshold", 0.5)
        )
        val_metrics = evaluate(model, val_set, device, batch_size, label_threshold, config)
        metrics = {"train/loss": float(np.mean(losses)) if losses else 0.0, **val_metrics}
        logger.log_metrics("train_gnn", epoch, metrics)
        checkpoint = build_checkpoint(
            config,
            global_step=epoch,
            epoch=epoch,
            metrics=metrics,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        manager.save_latest(checkpoint, "gnn")
        manager.save(f"gnn_epoch_{epoch:04d}.pt", checkpoint)
        if val_metrics["val/mean_f1"] >= best_f1:
            best_f1 = val_metrics["val/mean_f1"]
            manager.save("gnn_best_val_f1.pt", checkpoint)
            logger.event("best checkpoint updated")
    logger.close()


def evaluate(
    model,
    dataset,
    device: str,
    batch_size: int = 1,
    label_threshold: float = 0.5,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    predictions: list[list[int]] = [[] for _ in AFFORDANCE_NAMES]
    targets: list[list[int]] = [[] for _ in AFFORDANCE_NAMES]
    config = config or {}
    eval_workers = max(0, int(config.get("gnn_training", {}).get("eval_dataloader_workers", 0)))
    with torch.no_grad():
        for graph, labels, _example_count in make_graph_loader(
            dataset,
            config,
            batch_size,
            shuffle=False,
            epoch=0,
            num_workers=eval_workers,
        ):
            torch_graph = graph.to_torch(device)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            output = model(torch_graph)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.affordance_logits, target)
            losses.append(float(loss.cpu()))
            pred = (output.affordance_probs.detach().cpu().numpy() >= 0.5).astype(int)
            truth = (labels >= label_threshold).astype(int)
            for col in range(len(AFFORDANCE_NAMES)):
                predictions[col].extend(pred[:, col].tolist())
                targets[col].extend(truth[:, col].tolist())
    metrics = {"val/loss": float(np.mean(losses)) if losses else 0.0}
    f1s = []
    for idx, name in enumerate(AFFORDANCE_NAMES):
        f1 = binary_f1(predictions[idx], targets[idx])
        metrics[f"val/affordance_f1_{name}"] = f1
        f1s.append(f1)
    metrics["val/mean_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return metrics


def make_graph_loader(
    dataset: Any,
    config: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    epoch: int,
    num_workers: int | None = None,
):
    import torch
    from torch.utils.data import DataLoader

    train_cfg = config.get("gnn_training", {})
    workers = _dataloader_workers(config) if num_workers is None else max(0, int(num_workers))
    generator = torch.Generator()
    generator.manual_seed(int(config.get("project", {}).get("seed", 42)) + int(epoch))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": _collate_graph_examples,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(train_cfg.get("prefetch_factor", 2)))
        context = train_cfg.get("dataloader_multiprocessing_context")
        if context:
            kwargs["multiprocessing_context"] = str(context)
    return DataLoader(dataset, **kwargs)


def _collate_graph_examples(examples: list[tuple[Any, np.ndarray]]) -> tuple[Any, np.ndarray, int]:
    graph, labels = batch_graph_examples(examples)
    return graph, labels, len(examples)


def _dataloader_workers(config: dict[str, Any]) -> int:
    train_cfg = config.get("gnn_training", {})
    if not bool(train_cfg.get("use_dataloader", True)):
        return 0
    if "dataloader_workers" in train_cfg:
        return max(0, int(train_cfg.get("dataloader_workers", 0)))
    return max(0, int(config.get("hardware", {}).get("num_workers", 0)))


def build_positive_weight(dataset: Any, train_config: dict[str, Any], device: str, logger: RunLogger):
    import torch

    spec = train_config.get("positive_weight")
    if spec in (None, False, 0, "false", "none", "off"):
        return None
    if isinstance(spec, int | float):
        values = np.full((len(AFFORDANCE_NAMES),), float(spec), dtype=np.float32)
    elif isinstance(spec, list):
        values = np.asarray(spec, dtype=np.float32)
        if values.shape[0] != len(AFFORDANCE_NAMES):
            raise ValueError(f"positive_weight must have {len(AFFORDANCE_NAMES)} values.")
    elif str(spec).lower() == "auto":
        max_weight = max(1.0, float(train_config.get("positive_weight_max", 20.0)))
        cached_stats = _label_statistics(dataset, logger)
        if cached_stats is not None:
            positives, node_count = cached_stats
            values = _positive_weight_values(positives, node_count, max_weight)
            logger.event(
                "estimated positive weights from cache statistics "
                f"sample_nodes={node_count} "
                f"weights={dict(zip(AFFORDANCE_NAMES, [round(float(item), 3) for item in values]))}"
            )
            return torch.as_tensor(values, dtype=torch.float32, device=device)
        sample_rows = min(len(dataset), max(1, int(train_config.get("positive_weight_sample_rows", 2000))))
        indices = np.linspace(0, len(dataset) - 1, num=sample_rows, dtype=np.int64)
        positives = np.zeros((len(AFFORDANCE_NAMES),), dtype=np.float64)
        node_count = 0
        start = time.perf_counter()
        progress_every = max(1, int(train_config.get("positive_weight_log_interval_rows", 250)))
        logger.event(f"estimating positive weights sample_rows={sample_rows}")
        for sample_idx, idx in enumerate(indices, start=1):
            _, labels = dataset[int(idx)]
            positives += labels.sum(axis=0)
            node_count += labels.shape[0]
            if sample_idx == 1 or sample_idx % progress_every == 0 or sample_idx == sample_rows:
                elapsed = time.perf_counter() - start
                logger.event(
                    "estimating positive weights "
                    f"rows={sample_idx}/{sample_rows} sample_nodes={node_count} rows_per_sec={sample_idx / max(elapsed, 1e-9):.2f}"
                )
        values = _positive_weight_values(positives, node_count, max_weight)
        logger.event(
            "estimated positive weights "
            f"sample_rows={sample_rows} sample_nodes={node_count} elapsed_sec={time.perf_counter() - start:.1f} "
            f"weights={dict(zip(AFFORDANCE_NAMES, [round(float(item), 3) for item in values]))}"
        )
    else:
        raise ValueError("positive_weight must be auto, a number, a list, false, or none.")
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _label_statistics(dataset: Any, logger: RunLogger) -> tuple[np.ndarray, int] | None:
    current = dataset
    view_indices = None
    if hasattr(current, "indices") and hasattr(current, "dataset"):
        view_indices = list(current.indices)
        current = current.dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    if view_indices is not None and hasattr(current, "entries") and hasattr(current, "_load_shard"):
        return _cache_view_label_statistics(current, view_indices, logger)
    method = getattr(current, "label_statistics", None)
    if method is None:
        return None
    return method()


def _cache_view_label_statistics(cache_dataset: Any, indices: list[int], logger: RunLogger) -> tuple[np.ndarray, int]:
    by_shard: dict[str, list[int]] = {}
    for index in indices:
        entry = cache_dataset.entries[int(index)]
        by_shard.setdefault(str(entry["shard"]), []).append(int(entry["index"]))

    positives = np.zeros((len(AFFORDANCE_NAMES),), dtype=np.float64)
    node_count = 0
    start = time.perf_counter()
    total_shards = len(by_shard)
    logger.event(f"estimating positive weights from cache shards={total_shards} rows={len(indices)}")
    for shard_idx, (shard_name, local_indices) in enumerate(sorted(by_shard.items()), start=1):
        shard = cache_dataset._load_shard(shard_name)
        for local_index in local_indices:
            labels = np.asarray(shard[local_index]["labels"], dtype=np.float32)
            positives += labels.sum(axis=0)
            node_count += labels.shape[0]
        elapsed = time.perf_counter() - start
        logger.event(
            "estimating positive weights from cache "
            f"shards={shard_idx}/{total_shards} rows_seen={sum(len(v) for _, v in list(sorted(by_shard.items()))[:shard_idx])} "
            f"sample_nodes={node_count} shards_per_sec={shard_idx / max(elapsed, 1e-9):.2f}"
        )
    return positives, node_count


def _positive_weight_values(positives: np.ndarray, node_count: int, max_weight: float) -> np.ndarray:
    negatives = max(node_count, 1) - positives
    values = np.ones((len(AFFORDANCE_NAMES),), dtype=np.float32)
    present = positives > 0
    values[present] = np.clip(negatives[present] / positives[present], 1.0, max_weight).astype(np.float32)
    return values


if __name__ == "__main__":
    main()
