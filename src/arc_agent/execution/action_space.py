"""Shared low-level action constants."""

from __future__ import annotations

from enum import IntEnum


class GridAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    COMPLEX = 5
    UNDO = 6
    WAIT = 7


ACTION_DELTAS = {
    GridAction.UP: (0, -1),
    GridAction.DOWN: (0, 1),
    GridAction.LEFT: (-1, 0),
    GridAction.RIGHT: (1, 0),
}


def action_from_delta(dx: int, dy: int) -> GridAction:
    for action, delta in ACTION_DELTAS.items():
        if delta == (dx, dy):
            return action
    return GridAction.WAIT
