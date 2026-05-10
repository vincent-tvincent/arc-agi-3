# Training

Install the package from the repo root:

```bash
pip install -e .
pip install -e .[train]
```

The base install covers mock evaluation and tests. The `train` extra installs PyTorch and TensorBoard for neural training.

Command implementations live in `src/arc_agent/cli/`, with root `scripts/` wrappers kept for compatibility. Editable installs also expose `arc-agent-*` console commands.

## Collect Mock Rollouts

```bash
python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000
```

This writes JSONL rows under `/run/media/blue-lobster/disk3/CS274p_output/runs/gnn_mock_data/` by default. Each row has a frame, action, episode, step, and synthetic color-affordance labels.

## Collect Official ARC Rollouts

```bash
python src/experience_collection/collect_experience.py --game all --steps 200 --offline --out-dir /run/media/blue-lobster/disk3/CS274p_output/training_runs --training-out-dir /run/media/blue-lobster/disk3/CS274p_output/training_examples
```

This uses the official `arc_agi` environment files and writes raw transition logs to `/run/media/blue-lobster/disk3/CS274p_output/training_runs/` plus training-example JSONL files to `/run/media/blue-lobster/disk3/CS274p_output/training_examples/`.

## Train GNN

```bash
python scripts/train_gnn.py --config configs/train_gnn.yaml
```

For official ARC rollout data:

```bash
python scripts/train_gnn.py --config configs/train_gnn_arc.yaml
```

Official games do not expose ground-truth affordance labels, so this path trains on weak pseudo-labels derived from frame deltas, terminal states, and changed cells.

For large official datasets, `configs/train_gnn_arc.yaml` uses lazy JSONL indexing by default:

```yaml
gnn_training:
  lazy_index: true
  index_workers: 8
```

Set `lazy_index: false` to eagerly load all rows into memory. Increase `index_workers` if indexing many JSONL files is disk-bound and your storage can handle more parallel reads.

Checkpoints are written to:

```text
/run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/checkpoints/gnn_latest.pt
/run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/checkpoints/gnn_best_val_f1.pt
/run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/checkpoints/gnn_epoch_XXXX.pt
```

Logs include train loss, validation loss, per-affordance F1, mean F1, and GPU memory when CUDA is available.

## Train PPO

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --pretrained-gnn /run/media/blue-lobster/disk3/CS274p_output/runs/<run>/checkpoints/gnn_best_val_f1.pt
```

For the official ARC environment:

```bash
python scripts/train_ppo.py --config configs/train_ppo_arc.yaml --game ls20 --pretrained-gnn /run/media/blue-lobster/disk3/CS274p_output/runs/train_gnn_arc/checkpoints/gnn_best_val_f1.pt
```

Resume with:

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --resume /run/media/blue-lobster/disk3/CS274p_output/runs/<run>/checkpoints/ppo_latest.pt
```

PPO trains high-level strategy selection. The planner converts the selected strategy into a single low-level action and replans every step. With `--env arcagi3` or `configs/train_ppo_arc.yaml`, actions are sent through the official ARC adapter.

## Hardware Notes

`configs/hardware_3080_10gb.yaml` starts with conservative RTX 3080 defaults: mixed precision enabled, 8 PPO envs, 512 rollout steps, 512 minibatches, 3 GNN layers, and 128 hidden units. Active run artifacts stay in `/run/media/blue-lobster/disk3/CS274p_output/runs/`; large or older artifacts can be moved to the configured archive paths.
