"""A* pathfinding over grid cells."""

from __future__ import annotations

import heapq

from arc_agent.execution.action_space import ACTION_DELTAS, action_from_delta
from arc_agent.planning.bfs import Cell, in_bounds


def astar_path(
    start: Cell,
    goal: Cell,
    width: int,
    height: int,
    blocked: set[Cell] | None = None,
    risk: dict[Cell, float] | None = None,
    max_nodes: int = 20000,
) -> list[int] | None:
    blocked = blocked or set()
    risk = risk or {}
    if start == goal:
        return []
    if goal in blocked:
        return None
    frontier: list[tuple[float, int, Cell]] = [(heuristic(start, goal), 0, start)]
    parent: dict[Cell, Cell | None] = {start: None}
    cost_so_far: dict[Cell, float] = {start: 0.0}
    visited = 0
    while frontier and visited < max_nodes:
        _, _, current = heapq.heappop(frontier)
        visited += 1
        if current == goal:
            return reconstruct(parent, goal)
        for dx, dy in ACTION_DELTAS.values():
            nxt = (current[0] + dx, current[1] + dy)
            if not in_bounds(nxt, width, height) or nxt in blocked:
                continue
            new_cost = cost_so_far[current] + 1.0 + float(risk.get(nxt, 0.0)) * 5.0
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, visited, nxt))
                parent[nxt] = current
    return None


def heuristic(a: Cell, b: Cell) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def reconstruct(parent: dict[Cell, Cell | None], goal: Cell) -> list[int]:
    cells: list[Cell] = []
    cursor: Cell | None = goal
    while cursor is not None:
        cells.append(cursor)
        cursor = parent[cursor]
    cells.reverse()
    return [int(action_from_delta(b[0] - a[0], b[1] - a[1])) for a, b in zip(cells, cells[1:], strict=False)]
