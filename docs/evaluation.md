# Evaluation

Run the local mock baseline:

```bash
python scripts/evaluate_agent.py --config configs/debug_cpu.yaml --env mock --episodes 20
```

Artifacts are stored under:

```text
/run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/eval/
/run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/eval/replays/
/run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/logs/
```

Inspect a saved replay:

```bash
python scripts/replay_episode.py --replay /run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/eval/replays/episode_0000.json.gz
```

Inspect graph construction for a replay step:

```bash
python scripts/inspect_graphs.py --replay /run/media/blue-lobster/disk3/CS274p_output/runs/<run_name>/eval/replays/episode_0000.json.gz --step 10
```

## ARC-AGI-3 Adapter

The official adapter is available through:

```bash
python scripts/evaluate_agent.py --config configs/hardware_3080_10gb.yaml --env arcagi3 --game ls20 --episodes 5
```

It requires the official toolkit and project `.env` setup already used by the existing collector. The v1 adapter is intentionally thin; it does not add game-specific logic.

## No Game-Specific Hardcoding

The planner and candidate generator operate on generic affordance beliefs and graph relations. Mock color labels are only used for synthetic training and local smoke tests. Public game IDs are not hardcoded in the agent logic.
