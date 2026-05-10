"""Console, file, CSV, and optional TensorBoard logging."""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from arc_agent.utils.device import cuda_memory_gb


class RunLogger:
    """Shared run logger used by scripts and trainers."""

    def __init__(self, run_name: str, log_dir: str | Path, use_csv: bool = True, use_tensorboard: bool = False) -> None:
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter("[%(asctime)s][%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(self.log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream)
        self.logger.addHandler(file_handler)

        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8") if use_csv else None
        self.csv_writer: csv.DictWriter[str] | None = None
        self.tensorboard = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(str(self.log_dir.parent / "tensorboard"))
            except Exception:
                self.tensorboard = None

    def log_metrics(self, phase: str, step: int, metrics: dict[str, Any]) -> None:
        row = {"timestamp": datetime.utcnow().isoformat(), "phase": phase, "step": step, **metrics}
        row["gpu_memory_gb"] = round(cuda_memory_gb(), 4)
        parts = [f"{key}={value}" for key, value in row.items() if key not in {"timestamp", "phase"}]
        self.logger.info("[%s][step=%s] %s", phase, step, " ".join(parts))

        if self.csv_file is not None:
            if self.csv_writer is None:
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(row.keys()))
                if self.csv_file.tell() == 0:
                    self.csv_writer.writeheader()
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        if self.tensorboard is not None:
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    self.tensorboard.add_scalar(f"{phase}/{key}", value, step)

    def event(self, message: str) -> None:
        self.logger.info(message)

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
        if self.tensorboard is not None:
            self.tensorboard.close()
