# Troubleshooting

## PyTorch Missing

Mock evaluation and unit tests work without PyTorch. GNN and PPO training require:

```bash
pip install -e .[train]
```

If training scripts print a PyTorch requirement message, install the training extra in the environment you are using.

## Pytest Imports External Plugins

Some system Python installs auto-load third-party pytest plugins. The project disables napari auto-loading in `pyproject.toml` because that plugin may attempt to write cache files outside the workspace.

## No Training Data

Run:

```bash
python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000
```

Then rerun GNN training.

## Official ARC Environment Fails

Confirm the existing project `.env` is configured and that local games exist in `environment_files/`, or run the existing collector once in online mode to download environments.
