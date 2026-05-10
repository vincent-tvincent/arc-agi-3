# Data Format

## Mock Rollout JSONL

`scripts/collect_rollouts.py` writes one JSON object per observed frame:

```json
{
  "episode": 0,
  "step": 12,
  "frame": [[0, 1, 2]],
  "action": 3,
  "labels": {"1": ["controllable"], "2": ["blocking"]}
}
```

## Replay JSON

Evaluation replays are gzip-compressed JSON:

```json
{
  "episode": 0,
  "steps": [
    {
      "step": 0,
      "action": 1,
      "reward": -0.01,
      "done": false,
      "info": {"state": "NOT_FINISHED"},
      "frame": [[0, 1, 0]]
    }
  ]
}
```

## Graph Tensor Format

`GraphState` contains:

- `nodes`: tracked `ObjectComponent` records.
- `node_features`: shape `[num_nodes, node_feature_dim]`.
- `edge_index`: shape `[2, num_edges]`.
- `edge_attr`: shape `[num_edges, edge_feature_dim]`.
- `graph_features`: compact graph-level metadata.
- `action_context`: one-hot previous action vector.

Use `graph.to_torch(device)` before passing a graph to the GNN.

## Checkpoint Format

Checkpoints include model and optimizer state, full config, global step, epoch, metrics, RNG state, and git commit when available. PPO checkpoints use `actor_critic_state_dict`; GNN checkpoints use `model_state_dict`.

## Metrics CSV

`RunLogger` writes `runs/<run_name>/logs/metrics.csv` with timestamp, phase, step, scalar metrics, and GPU memory.
