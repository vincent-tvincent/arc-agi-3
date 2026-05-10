#!/usr/bin/env python3
"""Collect synthetic/mock rollout data for GNN pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from arc_agent.envs.mock_grid_env import MockGridEnv
from arc_agent.utils.config import load_config
from arc_agent.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_gnn.yaml")
    parser.add_argument("--env", default="mock", choices=["mock"])
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config.get("project", {}).get("seed", 42)))
    data_dir = Path(config.get("gnn_training", {}).get("data_dir", "/run/media/blue-lobster/disk3/CS274p_output/runs/gnn_mock_data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "mock_rollouts.jsonl"
    env = MockGridEnv(config.get("mock_env", {}))
    action_count = env.action_space_n
    rows_written = 0

    with out_path.open("w", encoding="utf-8") as file:
        for episode in range(args.episodes):
            obs = env.reset()
            for step in range(env.config.max_steps):
                action = (step + episode) % action_count
                file.write(
                    json.dumps(
                        {
                            "episode": episode,
                            "step": step,
                            "frame": obs.tolist(),
                            "action": action,
                            "labels": {str(k): sorted(v) for k, v in env.affordance_labels().items()},
                        }
                    )
                    + "\n"
                )
                rows_written += 1
                obs, _, done, _ = env.step(action)
                if done:
                    break
    print(f"wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()
