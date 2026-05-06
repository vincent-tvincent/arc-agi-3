"""Manually play a locally available ARC-AGI-3 environment."""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Literal, cast

import arc_agi
from arc_agi import OperationMode
from arc_agi.models import EnvironmentInfo
from arcengine import FrameDataRaw, GameAction, GameState

DEFAULT_ENVIRONMENTS_DIR = "src/environment_files"
DEFAULT_RECORDINGS_DIR = "src/recordings"
RenderMode = Literal["terminal", "terminal-fast", "human"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an ARC-AGI-3 game manually from the terminal.")
    parser.add_argument("env_id", nargs="?", help="Environment id, such as ls20, ls20/9607627b, or ls20-9607627b.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", default="terminal-fast", choices=["terminal", "terminal-fast", "human"])
    parser.add_argument("--online", action="store_true", help="Allow API access instead of local offline mode.")
    parser.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--no-recording", action="store_true")
    args = parser.parse_args()
    requested_env_id = cast(str | None, args.env_id)
    seed = cast(int, args.seed)
    render = cast(RenderMode, args.render)
    online = cast(bool, args.online)
    environments_dir = cast(str, args.environments_dir)
    recordings_dir = cast(str, args.recordings_dir)
    save_recording = not cast(bool, args.no_recording)

    mode = OperationMode.NORMAL if online else OperationMode.OFFLINE
    arcade = arc_agi.Arcade(
        operation_mode=mode,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
    )

    env_infos: list[EnvironmentInfo] = list(arcade.get_environments())
    if not env_infos:
        raise SystemExit(f"No environments found in {environments_dir!r}.")

    env_id = requested_env_id or prompt_for_environment(env_infos)
    game_id = resolve_environment_id(env_id, env_infos)
    env = arcade.make(game_id, seed=seed, render_mode=render, save_recording=save_recording)
    if env is None:
        raise SystemExit(f"Could not create environment {game_id!r}.")

    obs = env.observation_space
    if obs is None:
        obs = env.step(GameAction.RESET)

    print(f"Playing {game_id} with seed {seed}")
    print_help(env.action_space)

    step = 0
    while True:
        print_status(step, obs)
        raw = input("action> ").strip()
        if not raw:
            continue
        if raw.lower() in {"q", "quit", "exit"}:
            break
        if raw.lower() in {"h", "help", "?"}:
            print_help(env.action_space)
            continue
        if raw.lower() in {"r", "reset"}:
            obs = env.step(GameAction.RESET)
            step = 0
            continue

        try:
            action, data = parse_action(raw, env.action_space)
        except ValueError as error:
            print(f"Invalid input: {error}")
            continue

        obs = env.step(action, data=data)
        step += 1
        if obs and obs.state == GameState.WIN:
            print_status(step, obs)
            print("WIN")
            break

    print("Scorecard:", arcade.get_scorecard())


def prompt_for_environment(env_infos: Iterable[EnvironmentInfo]) -> str:
    infos = list(env_infos)
    print("Available environments:")
    for info in infos:
        print(f"  {info.game_id}  {getattr(info, 'title', '')}")
    return input("env_id> ").strip()


def resolve_environment_id(raw_env_id: str, env_infos: Iterable[EnvironmentInfo]) -> str:
    requested = raw_env_id.strip()
    if not requested:
        raise SystemExit("No environment id provided.")

    aliases: dict[str, str] = {}
    for info in env_infos:
        game_id = info.game_id
        aliases[game_id] = game_id
        aliases[game_id.replace("-", "/", 1)] = game_id
        aliases[game_id.split("-", 1)[0]] = game_id

    if requested in aliases:
        return aliases[requested]

    matches = [game_id for alias, game_id in aliases.items() if alias.startswith(requested)]
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise SystemExit(f"Ambiguous environment id {requested!r}: {', '.join(matches)}")
    raise SystemExit(f"Unknown environment id {requested!r}.")


def parse_action(raw: str, action_space: list[GameAction]) -> tuple[GameAction, dict[str, Any]]:
    parts = raw.replace(",", " ").split()
    token = parts[0].upper()
    if token.isdigit():
        token = f"ACTION{token}" if token != "0" else "RESET"

    try:
        action = GameAction[token]
    except KeyError as error:
        raise ValueError(f"unknown action {parts[0]!r}") from error

    if action not in action_space and action != GameAction.RESET:
        available = ", ".join(action.name for action in action_space)
        raise ValueError(f"{action.name} is not available. Available: {available}")

    if not action.is_complex():
        return action, {}

    if len(parts) != 3:
        raise ValueError(f"{action.name} needs coordinates: {action.name} x y")
    try:
        x, y = int(parts[1]), int(parts[2])
    except ValueError as error:
        raise ValueError("coordinates must be integers") from error
    return action, {"x": x, "y": y}


def print_help(action_space: list[GameAction]) -> None:
    print("Commands:")
    print("  q                 quit")
    print("  h                 help")
    print("  r                 reset")
    print("  1 / ACTION1       take a simple action")
    print("  ACTION6 x y       take a coordinate action")
    print("Available actions:")
    for action in action_space:
        suffix = " x y" if action.is_complex() else ""
        print(f"  {action.name}{suffix}")


def print_status(step: int, obs: FrameDataRaw | None) -> None:
    if obs is None:
        print(f"step={step} state=None")
        return
    print(
        f"step={step} state={obs.state.name} "
        f"levels_completed={obs.levels_completed}"
    )


if __name__ == "__main__":
    main()
