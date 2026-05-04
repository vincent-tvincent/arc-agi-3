"""Experience records for ARC-AGI-3 transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Transition:
    game_id: str
    step: int
    action: str
    action_data: dict[str, Any]
    before_hash: str
    after_hash: str
    before_frame: list[list[int]]
    after_frame: list[list[int]]
    changed_cells: list[dict[str, int]]
    state: str
    levels_completed: int
    available_actions: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

