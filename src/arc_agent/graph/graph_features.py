"""Feature-name helpers for object graphs."""

from __future__ import annotations


AFFORDANCE_NAMES = [
    "blocking",
    "movable",
    "collectible",
    "hazardous",
    "goal_related",
    "trigger",
    "controllable",
    "ui_or_status",
    "unknown_interactable",
]


def node_feature_names(max_colors: int, action_space_n: int, affordances: list[str] | None = None) -> list[str]:
    affordances = affordances or AFFORDANCE_NAMES
    names = ["x", "y", "width", "height", "area"]
    names.extend(f"color_{idx}" for idx in range(max_colors))
    names.extend(["motion_dx", "motion_dy", "changed", "new", "disappeared", "dist_to_agent", "touching_agent"])
    names.extend(f"belief_{name}" for name in affordances)
    names.extend(f"last_action_{idx}" for idx in range(action_space_n))
    return names


EDGE_FEATURE_NAMES = [
    "dx",
    "dy",
    "distance",
    "touching",
    "same_color",
    "same_row",
    "same_col",
    "temporal_same_track",
    "causal_probability",
    "reachable",
]
