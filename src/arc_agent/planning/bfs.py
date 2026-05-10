"""Breadth-first grid search."""

from __future__ import annotations

from collections import deque

from arc_agent.execution.action_space import ACTION_DELTAS, action_from_delta


Cell = tuple[int, int]


def bfs_path(start: Cell, goal: Cell, width: int, height: int, blocked: set[Cell]) -> list[int] | None:
    if start == goal:
        return []
    queue: deque[Cell] = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}
    while queue:
        cell = queue.popleft()
        for dx, dy in ACTION_DELTAS.values():
            nxt = (cell[0] + dx, cell[1] + dy)
            if not in_bounds(nxt, width, height) or nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cell
            if nxt == goal:
                return reconstruct_actions(parent, goal)
            queue.append(nxt)
    return None


def in_bounds(cell: Cell, width: int, height: int) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def reconstruct_actions(parent: dict[Cell, Cell | None], goal: Cell) -> list[int]:
    cells: list[Cell] = []
    cursor: Cell | None = goal
    while cursor is not None:
        cells.append(cursor)
        cursor = parent[cursor]
    cells.reverse()
    actions: list[int] = []
    for a, b in zip(cells, cells[1:], strict=False):
        actions.append(int(action_from_delta(b[0] - a[0], b[1] - a[1])))
    return actions
