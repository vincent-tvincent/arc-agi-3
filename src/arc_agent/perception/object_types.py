"""Core perception dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObjectComponent:
    id: int
    color_id: int | None
    pixels: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    frame_index: int
    track_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def cells(self) -> set[tuple[int, int]]:
        return {(int(x), int(y)) for x, y in self.pixels.tolist()}
