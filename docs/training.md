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

This writes JSONL rows under `runs/gnn_mock_data/` by default. Each row has a frame, action, episode, step, and synthetic color-affordance labels.

## Train GNN

```bash
python scripts/train_gnn.py --config configs/train_gnn.yaml
```

Checkpoints are written to:

```text
runs/<run_name>/checkpoints/gnn_latest.pt
runs/<run_name>/checkpoints/gnn_best_val_f1.pt
runs/<run_name>/checkpoints/gnn_epoch_XXXX.pt
```

Logs include train loss, validation loss, per-affordance F1, mean F1, and GPU memory when CUDA is available.

## Train PPO

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --pretrained-gnn runs/<run>/checkpoints/gnn_best_val_f1.pt
```

Resume with:

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --resume runs/<run>/checkpoints/ppo_latest.pt
```

PPO trains high-level strategy selection in the mock environment. The planner converts the selected strategy into a single low-level action and replans every step.

## Hardware Notes

`configs/hardware_3080_10gb.yaml` starts with conservative RTX 3080 defaults: mixed precision enabled, 8 PPO envs, 512 rollout steps, 512 minibatches, 3 GNN layers, and 128 hidden units. Active run artifacts stay in `runs/`; large or older artifacts can be moved to the configured archive paths.
