"""Batch collect training examples for all available ARC-AGI-3 games."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arc_agi
import yaml
from arc_agi import OperationMode

from collect_experience import DEFAULT_ENVIRONMENTS_DIR, DEFAULT_RECORDINGS_DIR, collect_game


DEFAULT_CONFIG: dict[str, Any] = {
    "steps": 200,
    "num_seeds": 5,
    "seed_start": 0,
    "offline": True,
    "game": "all",
    "out_dir": "training_runs",
    "training_out_dir": "training_examples",
    "render": None,
    "append": False,
    "environments_dir": DEFAULT_ENVIRONMENTS_DIR,
    "recordings_dir": DEFAULT_RECORDINGS_DIR,
}


@dataclass(frozen=True)
class BatchCollectConfig:
    steps: int
    num_seeds: int
    seed_start: int
    offline: bool
    game: str
    out_dir: str
    training_out_dir: str
    render: str | None
    append: bool
    environments_dir: str
    recordings_dir: str


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    mode = OperationMode.OFFLINE if config.offline else OperationMode.NORMAL
    arcade = arc_agi.Arcade(
        operation_mode=mode,
        environments_dir=config.environments_dir,
        recordings_dir=config.recordings_dir,
    )

    games = list(arcade.get_environments())
    selected = games if config.game == "all" else [game for game in games if game.game_id.startswith(config.game)]
    if not selected:
        available = ", ".join(game.game_id for game in games)
        raise SystemExit(f"No matching game for {config.game!r}. Available: {available}")

    out_dir = Path(config.out_dir)
    training_out_dir = Path(config.training_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_out_dir.mkdir(parents=True, exist_ok=True)

    seeds = range(config.seed_start, config.seed_start + config.num_seeds)
    print(f"collecting games={len(selected)} seeds={config.num_seeds} steps={config.steps}")
    print(f"raw runs -> {out_dir}")
    print(f"training examples -> {training_out_dir}")

    for seed in seeds:
        for game in selected:
            print(f"\ncollect game={game.game_id} seed={seed}")
            collect_game(
                arcade,
                game.game_id,
                config.steps,
                seed,
                out_dir,
                training_out_dir,
                config.render,
                config.append,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect training examples for all available games across many seeds.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    return parser.parse_args()


def load_config(path: Path) -> BatchCollectConfig:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file {path} must contain a YAML mapping.")

    unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown:
        raise SystemExit(f"Unknown config key(s) in {path}: {', '.join(unknown)}")

    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    validate_config(config)
    return BatchCollectConfig(**config)


def validate_config(config: dict[str, Any]) -> None:
    if int(config["steps"]) <= 0:
        raise SystemExit("steps must be greater than 0.")
    if int(config["num_seeds"]) <= 0:
        raise SystemExit("num_seeds must be greater than 0.")
    if int(config["seed_start"]) < 0:
        raise SystemExit("seed_start must be greater than or equal to 0.")
    if config["render"] not in {None, "terminal", "terminal-fast", "human"}:
        raise SystemExit("render must be null, terminal, terminal-fast, or human.")


if __name__ == "__main__":
    main()
