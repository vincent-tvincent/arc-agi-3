"""Collect ARC-AGI-3 transition data by probing public games."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

from arc3_pipeline.experience import Transition, transition_to_training_example
from arc3_pipeline.frame_utils import changed_cells, connected_components, frame_to_grid, grid_hash, background_color

DEFAULT_ENVIRONMENTS_DIR = "src/environment_files"
DEFAULT_RECORDINGS_DIR = "src/recordings"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20", help="Game id, or 'all' for every available game.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--training-out-dir", default="training_examples")
    parser.add_argument("--no-training-export", action="store_true", help="Do not write step-level training examples.")
    parser.add_argument("--render", default=None, choices=[None, "terminal", "terminal-fast", "human"])
    parser.add_argument("--offline", action="store_true", help="Use only already-downloaded local games.")
    parser.add_argument("--append", action="store_true", help="Append to an existing run file instead of replacing it.")
    parser.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    mode = OperationMode.OFFLINE if args.offline else OperationMode.NORMAL
    arc = arc_agi.Arcade(
        operation_mode=mode,
        environments_dir=args.environments_dir,
        recordings_dir=args.recordings_dir,
    )
    games = arc.get_environments()
    selected = games if args.game == "all" else [g for g in games if g.game_id.startswith(args.game)]
    if not selected:
        available = ", ".join(g.game_id for g in games)
        raise SystemExit(f"No matching game for {args.game!r}. Available: {available}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for game in selected:
        training_out_dir = None if args.no_training_export else Path(args.training_out_dir)
        collect_game(arc, game.game_id, args.steps, args.seed, out_dir, training_out_dir, args.render, args.append)


def collect_game(
    arc: arc_agi.Arcade,
    game_id: str,
    steps: int,
    seed: int,
    out_dir: Path,
    training_out_dir: Path | None,
    render: str | None,
    append: bool,
) -> None:
    env = arc.make(game_id, seed=seed, render_mode=render, save_recording=True)
    if env is None:
        print(f"skip {game_id}: could not create environment")
        return

    obs = env.observation_space
    if obs is None:
        obs = env.step(GameAction.RESET)
    if obs is None:
        print(f"skip {game_id}: no initial observation")
        return

    run_path = out_dir / f"{game_id.replace('/', '_')}_seed{seed}.jsonl"
    training_path = None
    if training_out_dir is not None:
        training_out_dir.mkdir(parents=True, exist_ok=True)
        training_path = training_out_dir / f"{game_id.replace('/', '_')}_seed{seed}.examples.jsonl"
    if run_path.exists() and not append:
        run_path.unlink()
    if training_path is not None and training_path.exists() and not append:
        training_path.unlink()
    for step in range(steps):
        before = frame_to_grid(obs.frame)
        before_levels_completed = obs.levels_completed
        action, data = choose_probe_action(env.action_space, before, step)
        next_obs = env.step(action, data=data)
        if next_obs is None:
            continue

        after = frame_to_grid(next_obs.frame)
        transition = Transition(
            game_id=game_id,
            step=step,
            action=action.name,
            action_data=data,
            before_hash=grid_hash(before),
            after_hash=grid_hash(after),
            before_frame=before,
            after_frame=after,
            changed_cells=changed_cells(before, after),
            state=next_obs.state.name,
            levels_completed=next_obs.levels_completed,
            available_actions=[a.name for a in env.action_space],
        )
        with run_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(transition.to_json()) + "\n")

        if training_path is not None:
            example = transition_to_training_example(transition, seed, before_levels_completed)
            with training_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(example.to_json()) + "\n")

        obs = next_obs
        if next_obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET) or next_obs
        if next_obs.state == GameState.WIN:
            break

    print(f"wrote {run_path}")
    if training_path is not None:
        print(f"wrote {training_path}")


def choose_probe_action(action_space: list[GameAction], grid: list[list[int]], step: int) -> tuple[GameAction, dict[str, Any]]:
    action = action_space[step % len(action_space)] if action_space else GameAction.RESET
    if action.is_complex():
        x, y = choose_coordinate_probe(grid, step)
        return action, {"x": x, "y": y}
    return action, {}


def choose_coordinate_probe(grid: list[list[int]], step: int) -> tuple[int, int]:
    bg = background_color(grid)
    components = connected_components(grid, ignore_color=bg)
    if components:
        centers = [tuple(component["center"]) for component in components[:20]]
        return centers[step % len(centers)]
    if not grid:
        return 0, 0
    return step % len(grid[0]), (step // max(1, len(grid[0]))) % len(grid)


if __name__ == "__main__":
    main()
