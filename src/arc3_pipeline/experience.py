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


@dataclass
class TrainingExample:
    example_id: str
    analysis_id: str
    game_id: str
    seed: int
    step: int
    input: dict[str, Any]
    action: dict[str, Any]
    target: dict[str, Any]
    outcome: dict[str, Any]
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def transition_to_training_example(
    transition: Transition,
    seed: int,
    before_levels_completed: int,
) -> TrainingExample:
    level_delta = transition.levels_completed - before_levels_completed
    return TrainingExample(
        example_id=f"{transition.game_id}_seed{seed}_step{transition.step}",
        analysis_id=f"{transition.game_id}_seed{seed}",
        game_id=transition.game_id,
        seed=seed,
        step=transition.step,
        input={
            "frame": transition.before_frame,
            "available_actions": transition.available_actions,
        },
        action={
            "name": transition.action,
            "data": transition.action_data,
        },
        target={
            "next_frame": transition.after_frame,
            "changed_cells": transition.changed_cells,
        },
        outcome={
            "state_after": transition.state,
            "levels_completed_before": before_levels_completed,
            "levels_completed_after": transition.levels_completed,
            "level_delta": level_delta,
            "is_win": transition.state == "WIN",
            "is_game_over": transition.state == "GAME_OVER",
            "changed_cell_count": len(transition.changed_cells),
            "frame_changed": transition.before_hash != transition.after_hash,
        },
        metadata={
            "before_hash": transition.before_hash,
            "after_hash": transition.after_hash,
        },
    )
