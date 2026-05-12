#!/usr/bin/env python3
"""Export visual GNN affordance predictions for one ARC frame."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.models.gnn_affordance import GNNAffordanceModel
from arc_agent.training.gnn_dataset import ARCTransitionGNNDataset, make_gnn_dataset
from arc_agent.utils.config import load_config, run_dir
from arc_agent.utils.device import select_device, torch_available


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


try:
    from arc_agi.rendering import COLOR_MAP as OFFICIAL_COLOR_MAP
except Exception:
    OFFICIAL_COLOR_MAP = {
        0: "#FFFFFFFF",
        1: "#CCCCCCFF",
        2: "#999999FF",
        3: "#666666FF",
        4: "#333333FF",
        5: "#000000FF",
        6: "#E53AA3FF",
        7: "#FF7BCCFF",
        8: "#F93C31FF",
        9: "#1E93FFFF",
        10: "#88D8F1FF",
        11: "#FFDC00FF",
        12: "#FF851BFF",
        13: "#921231FF",
        14: "#4FCC30FF",
        15: "#A356D6FF",
    }


PALETTE = np.asarray([_hex_to_rgb(OFFICIAL_COLOR_MAP[idx]) for idx in range(16)], dtype=np.uint8)


CATEGORY_COLORS = np.asarray(
    [
        [255, 89, 94],
        [25, 130, 196],
        [138, 201, 38],
        [255, 202, 58],
        [106, 76, 147],
        [255, 146, 76],
        [74, 222, 128],
        [244, 114, 182],
        [45, 212, 191],
    ],
    dtype=np.uint8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one frame of GNN affordance predictions.")
    parser.add_argument("--config", default="configs/train_gnn_arc.yaml")
    parser.add_argument("--checkpoint", default=None, help="Defaults to the run's gnn_best_val_f1.pt.")
    parser.add_argument("--data-dir", default=None, help="Override gnn_training.data_dir.")
    parser.add_argument("--data-format", default=None, choices=["auto", "mock", "arc", "official"])
    parser.add_argument("--game", default=None, help="Substring of a selected ARC jsonl filename, e.g. g50t or g50t-5849a774.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter when --game is used.")
    parser.add_argument("--row", type=int, default=0, help="Row in the selected file, or dataset index when --game is omitted.")
    parser.add_argument("--step", type=int, default=None, help="Select the first row with this transition step.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scale", type=int, default=16, help="Pixels per ARC cell in exported images.")
    parser.add_argument("--padding", type=int, default=1, help="Cell padding around exported object crops.")
    parser.add_argument("--out-dir", default=None, help="Defaults to <run_dir>/debug_gnn_frame.")
    args = parser.parse_args()

    if not torch_available():
        raise SystemExit("PyTorch is required for GNN debug inference.")
    import torch

    config = load_config(args.config)
    train_cfg = config.setdefault("gnn_training", {})
    train_cfg["lazy_index"] = True
    data_dir = Path(args.data_dir or train_cfg.get("data_dir", "runs/gnn_mock_data"))
    data_format = args.data_format or train_cfg.get("data_format", "auto")
    device = select_device(config.get("hardware", {}).get("device", "auto"))
    checkpoint_path = Path(args.checkpoint or (run_dir(config) / "checkpoints" / "gnn_best_val_f1.pt"))
    out_dir = Path(args.out_dir or (run_dir(config) / "debug_gnn_frame"))

    dataset, selected_index, source = _select_dataset(config, data_dir, data_format, args.game, args.seed, args.row, args.step)
    graph, labels = dataset[selected_index]
    frame = _frame_for_example(dataset, selected_index)

    model_cfg = config.get("model", {}).get("gnn", {})
    model = GNNAffordanceModel(
        node_in_dim=graph.node_feature_dim,
        edge_in_dim=graph.edge_feature_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        layers=int(model_cfg.get("layers", 3)),
        num_affordances=len(AFFORDANCE_NAMES),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(graph.to_torch(device))
    probs = output.affordance_probs.detach().cpu().numpy()
    predictions = probs >= float(args.threshold)

    export_dir = _unique_export_dir(out_dir, source, selected_index)
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in AFFORDANCE_NAMES:
        (export_dir / name).mkdir(exist_ok=True)

    write_png(export_dir / "frame.png", render_grid(frame, args.scale))
    write_png(export_dir / "frame_overlay.png", render_overlay(frame, graph.nodes, predictions, probs, args.scale))
    _write_predictions_csv(export_dir / "predictions.csv", graph, labels, probs, predictions)
    _write_metadata(export_dir / "metadata.json", args, source, selected_index, checkpoint_path, frame, graph)

    for node_idx, node in enumerate(graph.nodes):
        for category_idx, name in enumerate(AFFORDANCE_NAMES):
            if predictions[node_idx, category_idx]:
                prob = probs[node_idx, category_idx]
                crop = render_object_crop(frame, node, args.scale, args.padding, CATEGORY_COLORS[category_idx])
                crop_name = f"node_{node_idx:04d}_p{prob:.3f}_bbox_{node.bbox[0]}_{node.bbox[1]}_{node.bbox[2]}_{node.bbox[3]}.png"
                write_png(export_dir / name / crop_name, crop)

    print(f"exported GNN frame debug to {export_dir}")
    print(f"source={source} index={selected_index} nodes={graph.num_nodes} threshold={args.threshold}")


def _select_dataset(
    config: dict[str, Any],
    data_dir: Path,
    data_format: str,
    game: str | None,
    seed: int | None,
    row: int,
    step: int | None,
) -> tuple[Any, int, str]:
    if game:
        if not data_dir.is_dir():
            raise SystemExit(f"--game selection requires data_dir to be a directory: {data_dir}")
        matches = [path for path in sorted(data_dir.glob("*.jsonl")) if game in path.name]
        if seed is not None:
            seed_text = f"_seed{seed}"
            matches = [path for path in matches if seed_text in path.name]
        if not matches:
            raise SystemExit(f"No jsonl file in {data_dir} matched game={game!r} seed={seed!r}.")
        if len(matches) > 1:
            shown = "\n".join(str(path) for path in matches[:10])
            raise SystemExit(f"Multiple files matched. Use a more specific --game or --seed:\n{shown}")
        dataset = ARCTransitionGNNDataset(matches[0], config)
        index = _find_step_index(dataset, step) if step is not None else row
        return dataset, _check_index(dataset, index), matches[0].stem

    dataset = make_gnn_dataset(data_dir, config, data_format)
    index = _find_step_index(dataset, step) if step is not None else row
    return dataset, _check_index(dataset, index), data_dir.stem


def _check_index(dataset: Any, index: int) -> int:
    if index < 0 or index >= len(dataset):
        raise SystemExit(f"Selected row {index} is outside dataset length {len(dataset)}.")
    return index


def _find_step_index(dataset: Any, step: int | None) -> int:
    if step is None:
        return 0
    for index in range(len(dataset)):
        row = _raw_row(dataset, index)
        normalized = _normalize_any(dataset, row)
        if normalized is not None and int(normalized.get("step", -1)) == int(step):
            return index
    raise SystemExit(f"No row with step={step} found in selected data.")


def _raw_row(dataset: Any, index: int) -> dict[str, Any]:
    if isinstance(dataset, ARCTransitionGNNDataset):
        return dataset._read_entry(index) if dataset.lazy_index else dataset.rows[index]
    return dataset.rows[index]


def _normalize_any(dataset: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(dataset, ARCTransitionGNNDataset):
        return dataset._normalize_row(row)
    if "frame" in row:
        return {"step": int(row.get("step", 0)), "before_frame": row["frame"]}
    return None


def _frame_for_example(dataset: Any, index: int) -> np.ndarray:
    row = _raw_row(dataset, index)
    normalized = _normalize_any(dataset, row)
    if normalized is None or normalized.get("before_frame") is None:
        raise SystemExit(f"Could not find a frame for selected row {index}.")
    return np.asarray(normalized["before_frame"], dtype=np.int64)


def _unique_export_dir(root: Path, source: str, index: int) -> Path:
    safe_source = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in source)
    candidate = root / f"{safe_source}_row{index:06d}"
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        numbered = root / f"{safe_source}_row{index:06d}_{suffix}"
        if not numbered.exists():
            return numbered
        suffix += 1


def render_grid(frame: np.ndarray, scale: int) -> np.ndarray:
    colors = PALETTE[np.mod(frame, len(PALETTE))]
    return np.repeat(np.repeat(colors, scale, axis=0), scale, axis=1)


def render_overlay(frame: np.ndarray, nodes: list[Any], predictions: np.ndarray, probs: np.ndarray, scale: int) -> np.ndarray:
    image = to_grayscale(render_grid(frame, scale)).astype(np.float32)
    for node_idx, node in enumerate(nodes):
        active = np.flatnonzero(predictions[node_idx])
        if active.size == 0:
            continue
        best_category = int(active[np.argmax(probs[node_idx, active])])
        color = CATEGORY_COLORS[best_category].astype(np.float32)
        for x, y in node.cells:
            y0, y1 = y * scale, (y + 1) * scale
            x0, x1 = x * scale, (x + 1) * scale
            image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * 0.45 + color * 0.55
    return image.clip(0, 255).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    luminance = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    return np.repeat(luminance[..., None], 3, axis=2).astype(np.uint8)


def render_object_crop(frame: np.ndarray, node: Any, scale: int, padding: int, category_color: np.ndarray) -> np.ndarray:
    height, width = frame.shape
    x0 = max(0, int(node.bbox[0]) - padding)
    y0 = max(0, int(node.bbox[1]) - padding)
    x1 = min(width - 1, int(node.bbox[2]) + padding)
    y1 = min(height - 1, int(node.bbox[3]) + padding)
    crop = frame[y0 : y1 + 1, x0 : x1 + 1]
    image = render_grid(crop, scale).astype(np.float32)
    local_cells = {(x - x0, y - y0) for x, y in node.cells}
    mask = np.zeros(crop.shape, dtype=bool)
    for x, y in local_cells:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            mask[y, x] = True
    for y, x in np.argwhere(~mask):
        image[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale] *= 0.18
    for y, x in np.argwhere(mask):
        image[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale] = (
            image[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale] * 0.6 + category_color.astype(np.float32) * 0.4
        )
    return image.clip(0, 255).astype(np.uint8)


def _write_predictions_csv(path: Path, graph: Any, labels: np.ndarray, probs: np.ndarray, predictions: np.ndarray) -> None:
    fieldnames = [
        "node_index",
        "object_id",
        "track_id",
        "color_id",
        "bbox",
        "centroid",
        "area",
        "top_label",
        "top_label_confidence",
    ]
    for name in AFFORDANCE_NAMES:
        fieldnames.extend([f"prob_{name}", f"pred_{name}", f"label_{name}"])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for node_idx, node in enumerate(graph.nodes):
            top_label = ""
            top_label_confidence = 0.0
            if labels.size:
                top_idx = int(np.argmax(labels[node_idx]))
                top_label_confidence = float(labels[node_idx, top_idx])
                if top_label_confidence > 0.0:
                    top_label = AFFORDANCE_NAMES[top_idx]
            row = {
                "node_index": node_idx,
                "object_id": node.id,
                "track_id": node.track_id,
                "color_id": node.color_id,
                "bbox": " ".join(str(v) for v in node.bbox),
                "centroid": f"{node.centroid[0]:.3f} {node.centroid[1]:.3f}",
                "area": node.area,
                "top_label": top_label,
                "top_label_confidence": f"{top_label_confidence:.6f}",
            }
            for category_idx, name in enumerate(AFFORDANCE_NAMES):
                row[f"prob_{name}"] = f"{probs[node_idx, category_idx]:.6f}"
                row[f"pred_{name}"] = int(predictions[node_idx, category_idx])
                row[f"label_{name}"] = f"{float(labels[node_idx, category_idx]):.6f}" if labels.size else "0.000000"
            writer.writerow(row)


def _write_metadata(path: Path, args: argparse.Namespace, source: str, index: int, checkpoint_path: Path, frame: np.ndarray, graph: Any) -> None:
    data = {
        "source": source,
        "index": index,
        "checkpoint": str(checkpoint_path),
        "frame_shape": list(frame.shape),
        "num_nodes": graph.num_nodes,
        "threshold": args.threshold,
        "scale": args.scale,
        "categories": AFFORDANCE_NAMES,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}.")
    image = np.asarray(rgb, dtype=np.uint8)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


if __name__ == "__main__":
    main()
