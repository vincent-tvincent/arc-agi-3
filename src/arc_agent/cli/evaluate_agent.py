#!/usr/bin/env python3
"""Evaluate the object-centric agent on mock or ARC-AGI-3 environments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from arc_agent.agent import ObjectCentricAgent
from arc_agent.envs.arc_env_adapter import ArcAgi3EnvAdapter
from arc_agent.envs.mock_grid_env import COLOR_LABELS, MockGridEnv
from arc_agent.training.logging_utils import RunLogger
from arc_agent.utils.config import load_config, run_dir
from arc_agent.utils.io import write_json_gz
from arc_agent.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug_cpu.yaml")
    parser.add_argument("--env", default="mock", choices=["mock", "arcagi3"])
    parser.add_argument("--mode", default="heuristic", choices=["heuristic"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--offline", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config.get("project", {}).get("seed", 42)))
    root = run_dir(config)
    eval_dir = root / "eval"
    replay_dir = eval_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(config.get("project", {}).get("run_name", "eval"), root / "logs", config.get("logging", {}).get("use_csv_log", True), False)
    env = build_env(args.env, config, args.game, args.offline)
    color_map = COLOR_LABELS if args.env == "mock" else None

    rows: list[dict[str, float | int | bool]] = []
    successes: list[bool] = []
    returns: list[float] = []
    lengths: list[int] = []
    for episode in range(args.episodes):
        agent = ObjectCentricAgent(config, color_label_map=color_map)
        obs = env.reset()
        total_return = 0.0
        steps: list[dict[str, object]] = []
        done = False
        for step in range(config.get("mock_env", {}).get("max_steps", 80)):
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.observe_outcome(reward, done)
            total_return += reward
            steps.append({"step": step, "action": action, "reward": reward, "done": done, "info": info, "frame": obs.tolist()})
            obs = next_obs
            if done:
                break
        success = bool(done and total_return > 0)
        successes.append(success)
        returns.append(total_return)
        lengths.append(len(steps))
        rows.append({"episode": episode, "return": total_return, "success": success, "length": len(steps)})
        write_json_gz(replay_dir / f"episode_{episode:04d}.json.gz", {"episode": episode, "steps": steps})
        logger.log_metrics("eval", episode, {"return": round(total_return, 4), "success": float(success), "length": len(steps)})

    summary = {
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_return": sum(returns) / len(returns) if returns else 0.0,
        "mean_episode_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }
    write_metrics(eval_dir / "per_episode_metrics.csv", rows)
    write_json_gz(eval_dir / "summary.json.gz", summary)
    logger.event(f"summary {summary}")
    logger.close()


def build_env(kind: str, config: dict, game: str, offline: bool):
    if kind == "mock":
        return MockGridEnv(config.get("mock_env", {}))
    return ArcAgi3EnvAdapter(game_id=game, seed=int(config.get("project", {}).get("seed", 42)), offline=offline)


def write_metrics(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
