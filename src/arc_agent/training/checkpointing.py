"""Checkpoint save/resume helpers."""

from __future__ import annotations

import pickle
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from arc_agent.utils.device import torch_available


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch_available():
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def build_checkpoint(
    config: dict[str, Any],
    global_step: int,
    metrics: dict[str, Any] | None = None,
    epoch: int | None = None,
    **state: Any,
) -> dict[str, Any]:
    return {
        **state,
        "config": config,
        "global_step": global_step,
        "epoch": epoch,
        "metrics": metrics or {},
        "rng_state": rng_state(),
        "git_commit": git_commit(),
    }


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, keep_last: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, name: str, checkpoint: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / name
        if torch_available():
            import torch

            torch.save(checkpoint, path)
        else:
            with path.open("wb") as file:
                pickle.dump(checkpoint, file)
        if name.startswith("periodic_"):
            self._rotate_periodic()
        return path

    def save_latest(self, checkpoint: dict[str, Any], prefix: str) -> Path:
        return self.save(f"{prefix}_latest.pt", checkpoint)

    def save_best(self, checkpoint: dict[str, Any], prefix: str, metric_name: str) -> Path:
        return self.save(f"{prefix}_best_{metric_name}.pt", checkpoint)

    def load(self, path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
        checkpoint_path = Path(path)
        if torch_available():
            import torch

            return torch.load(checkpoint_path, map_location=map_location)
        with checkpoint_path.open("rb") as file:
            return pickle.load(file)

    def _rotate_periodic(self) -> None:
        periodic = sorted(self.checkpoint_dir.glob("periodic_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in periodic[:-self.keep_last]:
            old.unlink(missing_ok=True)
