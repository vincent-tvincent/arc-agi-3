"""Replay readers for evaluation and inspection scripts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def load_replay(path: str | Path) -> list[dict[str, Any]]:
    replay_path = Path(path)
    opener = gzip.open if replay_path.suffix == ".gz" else open
    with opener(replay_path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "steps" in data:
        return list(data["steps"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported replay format: {path}")
