#!/usr/bin/env python3
"""Train the GNN affordance model on synthetic/mock graph labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.models.gnn_affordance import GNNAffordanceModel
from arc_agent.training.checkpointing import CheckpointManager, build_checkpoint
from arc_agent.training.gnn_dataset import MockGNNDataset
from arc_agent.training.logging_utils import RunLogger
from arc_agent.training.metrics import binary_f1
from arc_agent.utils.config import load_config, run_dir
from arc_agent.utils.device import select_device, torch_available
from arc_agent.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_gnn.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if not torch_available():
        raise SystemExit("PyTorch is required for GNN training. Install with `pip install -e .[train]`.")
    import torch

    config = load_config(args.config)
    set_global_seed(int(config.get("project", {}).get("seed", 42)))
    device = select_device(config.get("hardware", {}).get("device", "auto"))
    root = run_dir(config)
    logger = RunLogger(config.get("project", {}).get("run_name", "train_gnn"), root / "logs", True, config.get("logging", {}).get("use_tensorboard", False))
    manager = CheckpointManager(root / "checkpoints", int(config.get("storage", {}).get("checkpoint_keep_last", 3)))
    dataset = MockGNNDataset(config.get("gnn_training", {}).get("data_dir", "runs/gnn_mock_data"), config)
    if len(dataset) == 0:
        raise SystemExit("No GNN training data found. Run `python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000` first.")
    train_set, val_set = dataset.split(float(config.get("gnn_training", {}).get("validation_fraction", 0.2)))
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

    epochs = int(config.get("gnn_training", {}).get("epochs", 10))
    for epoch in range(start_epoch, epochs):
        model.train()
        losses: list[float] = []
        for idx in range(len(train_set)):
            graph, labels = train_set[idx]
            torch_graph = graph.to_torch(device)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            output = model(torch_graph)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.affordance_logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_set, device)
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


def evaluate(model, dataset, device: str) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    predictions: list[list[int]] = [[] for _ in AFFORDANCE_NAMES]
    targets: list[list[int]] = [[] for _ in AFFORDANCE_NAMES]
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, labels = dataset[idx]
            torch_graph = graph.to_torch(device)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            output = model(torch_graph)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.affordance_logits, target)
            losses.append(float(loss.cpu()))
            pred = (output.affordance_probs.detach().cpu().numpy() >= 0.5).astype(int)
            truth = labels.astype(int)
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


if __name__ == "__main__":
    main()
