"""Frame-to-grid conversion for ARC-AGI style observations."""

from __future__ import annotations

from typing import Any

import numpy as np


def frame_to_grid(frame: Any) -> np.ndarray:
    """Return a 2D integer numpy array from common ARC frame shapes."""
    if frame is None:
        return np.zeros((0, 0), dtype=np.int64)
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if not frame:
        return np.zeros((0, 0), dtype=np.int64)
    if isinstance(frame[0], np.ndarray):
        frame = [layer.tolist() for layer in frame]
    if isinstance(frame[0][0], list):
        frame = frame[0]
    return np.asarray(frame, dtype=np.int64)


def grid_hash(grid: np.ndarray) -> str:
    import hashlib

    return hashlib.sha1(np.asarray(grid, dtype=np.int64).tobytes()).hexdigest()
