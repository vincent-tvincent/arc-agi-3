"""Collect ARC-AGI-3 transition data by probing public games."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

from arc3_pipeline.experience import Transition, transition_to_training_example
from arc3_pipeline.frame_utils import changed_cells, connected_components, frame_to_grid, grid_hash, background_color

DEFAULT_ENVIRONMENTS_DIR = "environment_files"
DEFAULT_OUTPUT_ROOT = "/run/media/blue-lobster/disk3/CS274p_output"
DEFAULT_RUNS_DIR = f"{DEFAULT_OUTPUT_ROOT}/training_runs"
DEFAULT_TRAINING_EXAMPLES_DIR = f"{DEFAULT_OUTPUT_ROOT}/training_examples"
DEFAULT_RECORDINGS_DIR = f"{DEFAULT_OUTPUT_ROOT}/recordings"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20", help="Game id, or 'all' for every available game.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action",
        help="Use a controlled action instead of the probe policy, e.g. ACTION1, 1, or 'ACTION6 3 5'.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--training-out-dir", default=DEFAULT_TRAINING_EXAMPLES_DIR)
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
        collect_game(
            arc,
            game.game_id,
            args.steps,
            args.seed,
            out_dir,
            training_out_dir,
            args.render,
            args.append,
            args.action,
        )


def collect_game(
    arc: arc_agi.Arcade,
    game_id: str,
    steps: int,
    seed: int,
    out_dir: Path,
    training_out_dir: Path | None,
    render: str | None,
    append: bool,
    controlled_action: str | None = None,
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
        if controlled_action is None:
            action, data = choose_probe_action(env.action_space, before, step)
        else:
            action, data = parse_controlled_action(controlled_action, env.action_space)
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


def parse_controlled_action(raw: str, action_space: list[GameAction]) -> tuple[GameAction, dict[str, Any]]:
    parts = raw.replace(",", " ").split()
    if not parts:
        raise SystemExit("--action cannot be empty.")

    token = parts[0].upper()
    if token.isdigit():
        token = f"ACTION{token}" if token != "0" else "RESET"

    try:
        action = GameAction[token]
    except KeyError as error:
        raise SystemExit(f"Unknown action {parts[0]!r}.") from error

    if action not in action_space and action != GameAction.RESET:
        available = ", ".join(candidate.name for candidate in action_space)
        raise SystemExit(f"{action.name} is not available. Available: {available}")

    if not action.is_complex():
        if len(parts) != 1:
            raise SystemExit(f"{action.name} does not take coordinates.")
        return action, {}

    if len(parts) != 3:
        raise SystemExit(f"{action.name} needs coordinates: --action '{action.name} x y'")
    try:
        x, y = int(parts[1]), int(parts[2])
    except ValueError as error:
        raise SystemExit("Coordinates must be integers.") from error
    return action, {"x": x, "y": y}


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
