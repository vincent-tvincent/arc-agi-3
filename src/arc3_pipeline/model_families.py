"""Bounded hypothesis generation for the seven starting model families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from .experience import Transition
from .frame_utils import background_color, connected_components

FAMILIES = [
    "navigation",
    "click_select",
    "toggle",
    "collection",
    "push_block",
    "sequencing",
    "pattern_transform",
]


@dataclass
class Hypothesis:
    family: str
    score: float
    confidence: str
    evidence: list[str]
    parameters: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def generate_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    hypotheses: list[Hypothesis] = []
    hypotheses.extend(_navigation_hypotheses(transitions))
    hypotheses.extend(_click_hypotheses(transitions))
    hypotheses.extend(_toggle_hypotheses(transitions))
    hypotheses.extend(_generic_family_hypotheses(transitions))
    return sorted(hypotheses, key=lambda h: h.score, reverse=True)


def _navigation_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    by_action: dict[str, list[tuple[int, int]]] = {}
    for transition in transitions:
        changes = transition.changed_cells
        if len(changes) != 2:
            continue
        removed = [c for c in changes if c["before"] != c["after"]]
        if len(removed) != 2:
            continue
        c1, c2 = removed
        if c1["before"] == c2["after"] and c1["after"] == c2["before"]:
            dx = c2["x"] - c1["x"]
            dy = c2["y"] - c1["y"]
            if abs(dx) + abs(dy) == 1:
                by_action.setdefault(transition.action, []).append((dx, dy))

    action_map: dict[str, str] = {}
    evidence: list[str] = []
    direction_names = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
    for action, deltas in by_action.items():
        dx = int(median([d[0] for d in deltas]))
        dy = int(median([d[1] for d in deltas]))
        direction = direction_names.get((dx, dy))
        if direction:
            action_map[action] = direction
            evidence.append(f"{action} repeatedly caused unit movement {direction}")

    if not action_map:
        return []

    return [
        Hypothesis(
            family="navigation",
            score=2.0 * len(action_map),
            confidence=_confidence(len(action_map), 4),
            evidence=evidence,
            parameters={"action_map": action_map},
        )
    ]


def _click_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    coordinate_actions: dict[str, int] = {}
    for transition in transitions:
        if {"x", "y"}.issubset(transition.action_data):
            coordinate_actions[transition.action] = coordinate_actions.get(transition.action, 0) + 1
    if not coordinate_actions:
        return []
    evidence = [f"{action} accepted x/y data {count} time(s)" for action, count in coordinate_actions.items()]
    return [
        Hypothesis(
            family="click_select",
            score=1.5 * sum(coordinate_actions.values()),
            confidence=_confidence(sum(coordinate_actions.values()), 8),
            evidence=evidence,
            parameters={"coordinate_actions": sorted(coordinate_actions)},
        )
    ]


def _toggle_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    toggles: list[str] = []
    for transition in transitions:
        if 1 <= len(transition.changed_cells) <= 8 and transition.before_hash != transition.after_hash:
            before_colors = sorted({c["before"] for c in transition.changed_cells})
            after_colors = sorted({c["after"] for c in transition.changed_cells})
            if before_colors != after_colors:
                toggles.append(transition.action)
    if not toggles:
        return []
    return [
        Hypothesis(
            family="toggle",
            score=float(len(toggles)),
            confidence=_confidence(len(toggles), 6),
            evidence=[f"small color/state changes seen after actions: {sorted(set(toggles))}"],
            parameters={"candidate_actions": sorted(set(toggles))},
        )
    ]


def _generic_family_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    if not transitions:
        return []

    first_grid = transitions[0].before_frame
    bg = background_color(first_grid)
    components = connected_components(first_grid, ignore_color=bg)
    component_summary = [
        {"color": c["color"], "size": c["size"], "bbox": c["bbox"], "center": c["center"]}
        for c in components[:30]
    ]
    progress_actions = sorted(
        {
            t.action
            for t in transitions
            if t.levels_completed > 0 or t.state in {"WIN", "GAME_OVER"}
        }
    )

    hypotheses: list[Hypothesis] = []
    for family in ("collection", "push_block", "sequencing", "pattern_transform"):
        score = 0.5
        evidence = [f"initial board has {len(components)} non-background components"]
        if progress_actions:
            score += 2.0
            evidence.append(f"progress/end-state actions observed: {progress_actions}")
        hypotheses.append(
            Hypothesis(
                family=family,
                score=score,
                confidence="low",
                evidence=evidence,
                parameters={
                    "background_color": bg,
                    "component_summary": component_summary,
                    "progress_actions": progress_actions,
                },
            )
        )
    return hypotheses


def _confidence(value: int, strong_at: int) -> str:
    if value >= strong_at:
        return "high"
    if value >= max(2, strong_at // 2):
        return "medium"
    return "low"

