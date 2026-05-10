# GNN + PPO Agent Architecture

The new `arc_agent` package is additive to the existing transition-analysis pipeline. It implements an object-centric interactive agent:

```text
frame -> segmenter -> tracker -> graph builder -> GNN affordances
      -> belief updater -> candidate generator -> planner -> PPO scorer -> executor
```

ARC-AGI-3 observations are turn-based grid frames, with a small action interface. The public docs describe games as 2D grid environments with actions 1-5, optional coordinate action 6, and undo action 7. The technical report frames the benchmark around exploration, modeling, goal-setting, and planning under action-efficiency pressure.

## Components

`perception/segmenter.py` converts a frame into same-color connected components. It does not use CNN detection in v1.

`perception/object_tracker.py` assigns stable track IDs using Hungarian matching when SciPy is available and nearest-neighbor matching otherwise.

`graph/graph_builder.py` creates PyTorch-Geometric-style tensors without requiring PyG: `node_features`, `edge_index`, and `edge_attr`.

`models/gnn_affordance.py` predicts affordances such as blocking, movable, collectible, hazardous, goal-related, trigger, controllable, UI/status, and unknown-interactable. These are behavior priors, not game-specific object classes.

`belief/belief_update.py` performs online adaptation without optimizer steps. It updates probabilities from environment feedback: failed movement increases blocking belief, disappearance after contact increases collectible belief, remote changes create causal edges, and rewards increase goal-related belief.

`planning/candidates.py` generates generic subgoals. `planning/planner.py` uses A* and configurable symbolic scoring. PPO only chooses high-level strategies; the planner still handles concrete movement and safety.

`envs/mock_grid_env.py` provides local mechanics for walls, collectibles, pushables, triggers/doors, hazards, and goals. `envs/arc_env_adapter.py` is the optional bridge to the official toolkit.

## Why Affordances

Colors and shapes do not have fixed meanings across ARC-AGI-3 games. A red object can be a wall in one task and a target in another. The GNN therefore predicts what an object can do or how it may matter, and the belief updater corrects those priors from direct evidence.

## Why High-Level PPO

PPO operates over compact graph and belief vectors, selecting generic strategies such as testing unknown objects or reaching a likely goal. It does not learn raw pixel-to-action reflexes in v1. This keeps learning focused on reusable exploration and planning preferences.
