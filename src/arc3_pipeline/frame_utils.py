"""Frame conversion, differencing, and object helpers."""

from __future__ import annotations

from collections import Counter, deque
from hashlib import sha1
from typing import Any

Grid = list[list[int]]
Cell = tuple[int, int]


def frame_to_grid(frame: Any) -> Grid:
    """Return the first frame layer as a plain 2D integer grid."""
    if frame is None:
        return []
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if not frame:
        return []
    if hasattr(frame[0], "tolist"):
        frame = [layer.tolist() for layer in frame]
    if isinstance(frame[0][0], list):
        return [[int(value) for value in row] for row in frame[0]]
    return [[int(value) for value in row] for row in frame]


def grid_hash(grid: Grid) -> str:
    return sha1(repr(grid).encode("utf-8")).hexdigest()


def changed_cells(before: Grid, after: Grid) -> list[dict[str, int]]:
    changes: list[dict[str, int]] = []
    height = min(len(before), len(after))
    width = min(len(before[0]) if before else 0, len(after[0]) if after else 0)
    for y in range(height):
        for x in range(width):
            if before[y][x] != after[y][x]:
                changes.append(
                    {
                        "x": x,
                        "y": y,
                        "before": before[y][x],
                        "after": after[y][x],
                    }
                )
    return changes


def color_counts(grid: Grid) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    return dict(counts)


def background_color(grid: Grid) -> int | None:
    counts = color_counts(grid)
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def connected_components(grid: Grid, ignore_color: int | None = None) -> list[dict[str, Any]]:
    """Return 4-connected same-color components with bounding boxes and centers."""
    if not grid:
        return []
    height, width = len(grid), len(grid[0])
    seen: set[Cell] = set()
    components: list[dict[str, Any]] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in seen or grid[y][x] == ignore_color:
                continue
            color = grid[y][x]
            queue: deque[Cell] = deque([(x, y)])
            seen.add((x, y))
            cells: list[Cell] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and (nx, ny) not in seen
                        and grid[ny][nx] == color
                    ):
                        seen.add((nx, ny))
                        queue.append((nx, ny))

            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "center": [round(sum(xs) / len(xs)), round(sum(ys) / len(ys))],
                    "cells": cells,
                }
            )
    return components

