"""Deterministic connected-component segmentation for grid observations."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from arc_agent.perception.grid_extractor import frame_to_grid
from arc_agent.perception.object_types import ObjectComponent


@dataclass
class SegmenterConfig:
    min_component_area: int = 1
    background_color_mode: str = "auto"
    ignore_background: bool = True


class GridSegmenter:
    def __init__(self, config: dict[str, Any] | SegmenterConfig | None = None) -> None:
        if isinstance(config, SegmenterConfig):
            self.config = config
        else:
            data = config or {}
            self.config = SegmenterConfig(
                min_component_area=int(data.get("min_component_area", 1)),
                background_color_mode=str(data.get("background_color_mode", "auto")),
                ignore_background=bool(data.get("ignore_background", True)),
            )

    def background_color(self, grid: np.ndarray) -> int | None:
        if grid.size == 0:
            return None
        if self.config.background_color_mode != "auto":
            return int(self.config.background_color_mode)
        counts = Counter(int(value) for value in grid.reshape(-1).tolist())
        return counts.most_common(1)[0][0]

    def segment(self, frame: Any, frame_index: int = 0) -> list[ObjectComponent]:
        grid = frame_to_grid(frame)
        if grid.size == 0:
            return []
        height, width = grid.shape
        background = self.background_color(grid) if self.config.ignore_background else None
        seen: set[tuple[int, int]] = set()
        objects: list[ObjectComponent] = []
        component_id = 0

        for y in range(height):
            for x in range(width):
                color = int(grid[y, x])
                if (x, y) in seen or color == background:
                    continue
                cells = self._component_cells(grid, x, y, seen)
                if len(cells) < self.config.min_component_area:
                    continue
                xs = [cell[0] for cell in cells]
                ys = [cell[1] for cell in cells]
                pixels = np.asarray(cells, dtype=np.int64)
                objects.append(
                    ObjectComponent(
                        id=component_id,
                        color_id=color,
                        pixels=pixels,
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                        centroid=(float(sum(xs) / len(xs)), float(sum(ys) / len(ys))),
                        area=len(cells),
                        frame_index=frame_index,
                    )
                )
                component_id += 1
        return objects

    def _component_cells(
        self,
        grid: np.ndarray,
        start_x: int,
        start_y: int,
        seen: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        height, width = grid.shape
        color = int(grid[start_y, start_x])
        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
        seen.add((start_x, start_y))
        cells: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            cells.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen and int(grid[ny, nx]) == color:
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return cells
