# HDBSCAN Rate Feature Reference

The HDBSCAN vector builder aggregates each `training_examples/*.examples.jsonl` file into one numeric vector. Rate features are fractions in `[0, 1]`.

Implementation source: `src/hdbscan_pipeline/vector_builder.py`.

```text
rate = numerator / denominator
```

If the denominator is `0`, the code returns `0.0`.

## Whole-Run Rates

These rates are computed across every transition in one game/run example file.

| Feature | Numerator | Denominator | Meaning |
| --- | --- | --- | --- |
| `coordinate_action_rate` | Steps whose `action.data` contains both `x` and `y` | Total steps | How often the run used coordinate/click-style actions. |
| `coordinate_action_unique_xy_rate` | Unique `(x, y)` pairs used by coordinate actions | Coordinate-action steps | Whether coordinate actions explored many board locations or repeated the same locations. |
| `frame_changed_rate` | Steps where the frame changed after the action | Total steps | How often actions visibly affected the board. |
| `no_change_rate` | Steps where the frame did not change | Total steps | How often actions produced no visible board effect. Implemented as `1 - frame_changed_rate`. |
| `small_change_rate_1_8` | Steps with `1 <= changed_cell_count <= 8` | Total steps | Frequency of small local changes, such as movement, toggles, selection, or small interactions. |
| `medium_change_rate_9_32` | Steps with `9 <= changed_cell_count <= 32` | Total steps | Frequency of moderate board changes, such as area effects, multi-cell movement, or object transforms. |
| `large_change_rate_33_plus` | Steps with `changed_cell_count >= 33` | Total steps | Frequency of large/global changes, such as resets, level transitions, broad transforms, or animations. |
| `unique_before_hash_rate` | Unique `before_hash` values | Total steps | How many distinct states the collector visited before acting. |
| `unique_after_hash_rate` | Unique `after_hash` values | Total steps | How many distinct states resulted after actions. |
| `repeat_after_hash_rate` | Repeated `after_hash` states | Total steps | How often actions led to already-seen result states. Implemented as `1 - unique_after_hash_rate`. |
| `progress_rate` | Steps with `level_delta > 0` | Total steps | How often an action completed progress, usually level completion. |
| `game_over_rate` | Steps where `outcome.is_game_over` is true | Total steps | How often actions reached terminal states. |
| `win_rate` | Steps where `outcome.is_win` is true | Total steps | How often actions reached winning states. |

## Per-Action Rates

These features are generated for each action in:

```text
RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7
```

The actual feature names are built as `<ACTION>_<suffix>`, for example `ACTION1_used_rate` or `ACTION6_win_rate`.

| Feature Pattern | Numerator | Denominator | Meaning |
| --- | --- | --- | --- |
| `<ACTION>_used_rate` | Steps using that action | Total steps | How often the collector selected that action. |
| `<ACTION>_frame_changed_rate` | Steps using that action where the frame changed | Steps using that action | Whether this action tends to visibly affect the board. |
| `<ACTION>_progress_rate` | Steps using that action with `level_delta > 0` | Steps using that action | Whether this action is associated with level progress. |
| `<ACTION>_game_over_rate` | Steps using that action where `outcome.is_game_over` is true | Steps using that action | Whether this action tends to end the game. |
| `<ACTION>_win_rate` | Steps using that action where `outcome.is_win` is true | Steps using that action | Whether this action is directly associated with winning. |

## ACTION6 Coordinate Rates

`ACTION6` is treated specially because ARC-AGI-3 complex/click actions commonly appear as `ACTION6` with `x` and `y` payload data.

| Feature | Numerator | Denominator | Meaning |
| --- | --- | --- | --- |
| `ACTION6_coordinate_rate` | `ACTION6` steps whose `action.data` contains both `x` and `y` | Total `ACTION6` steps | Whether `ACTION6` is being used as a coordinate/click action. |
| `ACTION6_unique_xy_rate` | Unique `(x, y)` pairs used by coordinate `ACTION6` steps | Coordinate `ACTION6` steps | Whether `ACTION6` explores many board positions or repeats locations. |

## Related Fraction Feature

This feature is not named `rate`, but it is also computed with the same rate helper.

| Feature | Numerator | Denominator | Meaning |
| --- | --- | --- | --- |
| `initial_largest_component_fraction` | Cell count of the largest initial connected non-background component | Total cells in the initial frame | How much of the starting board is occupied by the largest object-like component. |

