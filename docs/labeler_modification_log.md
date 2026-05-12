# GNN Pseudo-Labeler Modification Log

This file records every intentional change to the ARC transition pseudo-labeler
used by `ARCTransitionGNNDataset`. Update this document whenever labeler behavior
changes, especially when a change affects generated GNN training labels.

## 2026-05-11 - LS20 role-disambiguation pass

Changed file:

- `src/arc_agent/training/gnn_dataset.py`
- `configs/train_gnn_arc.yaml`
- `tests/test_arc_transition_dataset.py`

Reason:

- LS20 debug export showed the status bar was recognized, but several gameplay
  roles were confused:
  - movable/player object predicted as `unknown_interactable`
  - life counter predicted as `unknown_interactable`
  - interactable cross/button predicted as `unknown_interactable`
  - goal target/gate predicted as `trigger`

Policy changes:

- Bottom HUD-like components are now labeled `ui_or_status` and excluded from
  interaction labels when `pseudo_labeling.hud_like_is_status` is true.
- `unknown_interactable` is now a fallback label when
  `pseudo_labeling.unknown_interactable_as_fallback` is true. A changed object is
  labeled unknown only if no stronger affordance label matched.
- Short, coherent object motion is still labeled `movable`, and compact moving
  objects can be labeled `controllable`.
- Coherent motion larger than `pseudo_labeling.controllable_max_step_distance`
  plus remote frame changes is treated as `goal_related` target motion instead
  of direct player control.
- Trigger labels are restricted to compact stationary source objects/buttons via
  `pseudo_labeling.trigger_source_max_area`,
  `pseudo_labeling.trigger_source_max_width`, and
  `pseudo_labeling.trigger_source_max_height`.
- Moving target/gate components are no longer labeled `trigger` just because
  they changed while other cells changed remotely.

Expected LS20 effect:

- Status bar: remains `ui_or_status`.
- Life counter and lower-left indicator: should become `ui_or_status`, not
  `unknown_interactable`.
- Interactable cross/button pieces: should become `trigger` when they are the
  compact stationary source of a remote change.
- Goal target/gate: should become `goal_related` during remote target shifts,
  not `trigger` or `controllable`.
- Player-like short motion: should become `movable` and `controllable`, not
  `unknown_interactable`.

Validation added:

- Tests for bottom HUD UI labeling.
- Tests for stationary remote controls becoming `trigger`.
- Tests for remote target shifts becoming `goal_related`, not `trigger`.
- Existing win/collectible tests updated so `unknown_interactable` is not
  applied when stronger labels are present.

## 2026-05-12 - HUD preview-container inheritance

Changed file:

- `src/arc_agent/training/gnn_dataset.py`
- `configs/train_gnn_arc.yaml`
- `tests/test_arc_transition_dataset.py`

Reason:

- LS20 target-shape indicator previews can contain colored pieces that look like
  small movable objects. Because connected-component segmentation separates the
  inner preview shape from its surrounding HUD panel, those inner pieces could
  be labeled `movable` when the preview changed.

Policy changes:

- Added HUD-container inheritance. Components nested inside an edge-positioned
  preview/container panel are now treated as `ui_or_status` before motion rules
  run.
- Added configurable controls:
  - `pseudo_labeling.hud_container_inheritance`
  - `pseudo_labeling.hud_container_padding`
  - `pseudo_labeling.hud_preview_use_position_filter`
  - `pseudo_labeling.hud_preview_edge_margin_ratio`
  - `pseudo_labeling.hud_preview_min_size`
  - `pseudo_labeling.hud_preview_max_area_ratio`

Expected LS20 effect:

- The target-shape preview pieces shown inside HUD indicators near any frame
  edge should be labeled `ui_or_status`, not `movable` or `controllable`.

Validation added:

- Test for an L-shaped target preview nested inside an edge-positioned HUD
  panel; the inner shape is labeled `ui_or_status` and not `movable`.
- Test that the same nested shape in a center-field container is not treated as
  HUD, reducing false positives on game-world objects.

## 2026-05-12 - Configurable soft pseudo-labels

Changed file:

- `src/arc_agent/training/gnn_dataset.py`
- `src/arc_agent/cli/train_gnn.py`
- `src/arc_agent/cli/debug_gnn_frame.py`
- `configs/train_gnn_arc.yaml`
- `tests/test_arc_transition_dataset.py`

Reason:

- Hard 0/1 pseudo-labels were too brittle for ambiguous ARC objects. Some
  objects have weak visual evidence for multiple roles, such as HUD preview
  pieces that visually move but should mostly be UI, or goal targets that shift
  remotely but are not player-controlled.

Policy changes:

- Added `pseudo_labeling.soft_labels.enabled`. When disabled, the labeler keeps
  hard 0/1 behavior. When enabled, rule gates emit confidence values in `[0, 1]`.
- Added configurable confidence values for status UI, HUD UI, HUD previews,
  win/goal objects, hazards, collectibles, short motion, controllable motion,
  remote targets, triggers, blocking, and unknown fallback.
- Added proportional confidence modifiers:
  - smallness for controllable, collectible, and trigger confidence
  - compactness for collectible and trigger confidence
  - largeness for blocking confidence
  - motion-step score for short coherent motion confidence
- Added `similar_max_motion_overlap_ratio` so overlapping sprite movement can be
  recognized while near-identical in-place flicker is rejected.
- Validation F1 now thresholds soft labels with
  `soft_labels.metric_label_threshold`; BCE training still uses raw soft labels.
- GNN debug `predictions.csv` now writes floating label confidences and includes
  `top_label` plus `top_label_confidence`.

Expected effect:

- A square player-like object with short coherent motion receives high but not
  absolute `movable` and `controllable` confidence.
- HUD preview pieces receive high `ui_or_status` confidence with small residual
  `movable`/`unknown_interactable` confidence.
- Remote targets receive strong `goal_related` confidence and only weak motion
  confidence.

Validation added:

- Test for soft square-agent labels scaling by motion and size.
- Test for HUD preview residual motion confidence.
- Existing hard-label tests still run with default hard mode.
