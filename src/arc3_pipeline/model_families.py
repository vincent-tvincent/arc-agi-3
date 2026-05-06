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
    # Evidence used for final ranking: each hypothesis has a numeric score
    # assigned by its detector. Higher score means that detector saw more direct
    # transition evidence for its family.
    return sorted(hypotheses, key=lambda h: h.score, reverse=True)


def _navigation_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    by_action: dict[str, list[tuple[int, int]]] = {}
    for transition in transitions:
        changes = transition.changed_cells
        # Evidence: a one-cell move usually changes exactly two cells: the
        # source cell and the destination cell.
        if len(changes) != 2:
            continue
        removed = [c for c in changes if c["before"] != c["after"]]
        # Evidence: both changed cells must have real before/after differences.
        if len(removed) != 2:
            continue
        c1, c2 = removed
        # Evidence: if the two cells swap colors, an object may have moved from
        # one cell into the adjacent cell while the old location changed back.
        if c1["before"] == c2["after"] and c1["after"] == c2["before"]:
            dx = c2["x"] - c1["x"]
            dy = c2["y"] - c1["y"]
            # Evidence: navigation is restricted here to cardinal one-step
            # motion, so exactly one axis should change by one cell.
            if abs(dx) + abs(dy) == 1:
                by_action.setdefault(transition.action, []).append((dx, dy))

    action_map: dict[str, str] = {}
    evidence: list[str] = []
    direction_names = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
    for action, deltas in by_action.items():
        # Evidence aggregation: use the median observed delta so a few noisy
        # transitions do not dominate the inferred direction.
        dx = int(median([d[0] for d in deltas]))
        dy = int(median([d[1] for d in deltas]))
        direction = direction_names.get((dx, dy))
        if direction:
            action_map[action] = direction
            evidence.append(f"{action} repeatedly caused unit movement {direction}")

    # Evidence threshold: without at least one action-to-direction mapping,
    # there is no concrete navigation hypothesis to return.
    if not action_map:
        return []

    return [
        Hypothesis(
            family="navigation",
            # Score evidence: each inferred movement action contributes equally.
            score=2.0 * len(action_map),
            confidence=_confidence(len(action_map), 4),
            evidence=evidence,
            parameters={"action_map": action_map},
        )
    ]


def _click_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    coordinate_actions: dict[str, int] = {}
    for transition in transitions:
        # Evidence: coordinate payloads mean the action can target a specific
        # board location, which is the core signal for click/select behavior.
        if {"x", "y"}.issubset(transition.action_data):
            coordinate_actions[transition.action] = coordinate_actions.get(transition.action, 0) + 1
    # Evidence threshold: no coordinate actions means no click/select evidence.
    if not coordinate_actions:
        return []
    evidence = [f"{action} accepted x/y data {count} time(s)" for action, count in coordinate_actions.items()]
    return [
        Hypothesis(
            family="click_select",
            # Score evidence: more coordinate-action transitions increase
            # confidence that coordinates are central to this game family.
            score=1.5 * sum(coordinate_actions.values()),
            confidence=_confidence(sum(coordinate_actions.values()), 8),
            evidence=evidence,
            parameters={"coordinate_actions": sorted(coordinate_actions)},
        )
    ]


def _toggle_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    toggles: list[str] = []
    for transition in transitions:
        # Evidence: toggles usually affect a small local region. Zero changed
        # cells is no effect; more than eight cells is treated as too broad for
        # this simple toggle detector.
        if 1 <= len(transition.changed_cells) <= 8 and transition.before_hash != transition.after_hash:
            before_colors = sorted({c["before"] for c in transition.changed_cells})
            after_colors = sorted({c["after"] for c in transition.changed_cells})
            # Evidence: the local region must actually change state/color, not
            # merely report positions with the same color set.
            if before_colors != after_colors:
                toggles.append(transition.action)
    # Evidence threshold: no small local color changes means no toggle evidence.
    if not toggles:
        return []
    return [
        Hypothesis(
            family="toggle",
            # Score evidence: each toggle-like transition contributes one point.
            score=float(len(toggles)),
            confidence=_confidence(len(toggles), 6),
            evidence=[f"small color/state changes seen after actions: {sorted(set(toggles))}"],
            parameters={"candidate_actions": sorted(set(toggles))},
        )
    ]


def _generic_family_hypotheses(transitions: list[Transition]) -> list[Hypothesis]:
    # Evidence threshold: generic object-family hypotheses need at least one
    # transition so we can inspect the initial board and progress signals.
    if not transitions:
        return []

    first_grid = transitions[0].before_frame
    # Evidence: the most common color is treated as background so connected
    # components can represent foreground objects or UI/game pieces.
    bg = background_color(first_grid)
    # Evidence: non-background connected components are object candidates for
    # collection, pushing, sequence elements, or pattern parts.
    components = connected_components(first_grid, ignore_color=bg)
    component_summary = [
        {"color": c["color"], "size": c["size"], "bbox": c["bbox"], "center": c["center"]}
        for c in components[:30]
    ]
    # Evidence: actions that complete levels, win, or end the game are likely
    # causally important even before we know the exact family.
    progress_actions = sorted(
        {
            t.action
            for t in transitions
            if t.levels_completed > 0 or t.state in {"WIN", "GAME_OVER"}
        }
    )

    hypotheses: list[Hypothesis] = []
    for family in ("collection", "push_block", "sequencing", "pattern_transform"):
        # Score evidence: base score means "there are object candidates, but no
        # specialized detector has confirmed this family yet."
        score = 0.5
        evidence = [f"initial board has {len(components)} non-background components"]
        # Score evidence: progress/end-state actions make the generic family
        # more plausible because they identify actions tied to game outcome.
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
