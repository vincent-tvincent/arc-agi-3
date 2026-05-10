"""Manually play a locally available ARC-AGI-3 environment."""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

import arc_agi
from arc_agi import OperationMode
from arc_agi.models import EnvironmentInfo
from arcengine import FrameDataRaw, GameAction, GameState

from arc3_pipeline.frame_utils import frame_to_grid

DEFAULT_ENVIRONMENTS_DIR = "environment_files"
DEFAULT_RECORDINGS_DIR = "/run/media/blue-lobster/disk3/CS274p_output/recordings"
DEFAULT_KEYMAP = Path(__file__).with_name("manual_play_keymap.yaml")
RenderMode = Literal["terminal", "terminal-fast", "human", None]

ARC_COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
    10: "#FFFFFF",
    11: "#6B5B95",
    12: "#88B04B",
    13: "#F7CAC9",
    14: "#92A8D1",
    15: "#955251",
}


@dataclass(frozen=True)
class KeymapConfig:
    keys: dict[str, GameAction]
    mouse_action: GameAction | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an ARC-AGI-3 game with keyboard and mouse controls.")
    parser.add_argument("env_id", nargs="?", help="Environment id, such as ls20, ls20/9607627b, or ls20-9607627b.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", default=None, choices=[None, "terminal", "terminal-fast", "human"])
    parser.add_argument("--online", action="store_true", help="Allow API access instead of local offline mode.")
    parser.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--no-recording", action="store_true")
    parser.add_argument("--keymap", default=str(DEFAULT_KEYMAP), help="YAML or JSON keymap for non-complex actions.")
    parser.add_argument("--cell-size", type=int, default=12, help="Pixel size used to draw one grid cell.")
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

    keymap = load_keymap(Path(cast(str, args.keymap)), env.action_space)
    print(f"Playing {game_id} with seed {seed}")
    print_help(env.action_space, keymap)

    try:
        window = ManualPlayWindow(
            env=env,
            arcade=arcade,
            initial_obs=obs,
            keymap=keymap,
            cell_size=max(1, cast(int, args.cell_size)),
        )
    except tk.TclError as error:
        raise SystemExit("The manual playground requires a graphical display for mouse input.") from error
    window.run()


class ManualPlayWindow:
    def __init__(
        self,
        env: Any,
        arcade: arc_agi.Arcade,
        initial_obs: FrameDataRaw | None,
        keymap: KeymapConfig,
        cell_size: int,
    ) -> None:
        self.env = env
        self.arcade = arcade
        self.obs = initial_obs
        self.keymap = keymap
        self.cell_size = cell_size
        self.step = 0

        self.root = tk.Tk()
        self.root.title("ARC-AGI-3 Playground")
        self.status = tk.StringVar()
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack()
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")

        self.root.bind("<Key>", self.on_key)
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.draw()

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        print("Scorecard:", self.arcade.get_scorecard())
        self.root.destroy()

    def on_key(self, event: tk.Event) -> None:
        key = str(event.keysym)
        if key in {"q", "Q", "Escape"}:
            self.close()
            return
        if key in self.keymap.keys:
            self.apply_action(self.keymap.keys[key], {})
            return
        if key.lower() in self.keymap.keys:
            self.apply_action(self.keymap.keys[key.lower()], {})

    def on_click(self, event: tk.Event) -> None:
        if self.keymap.mouse_action is None:
            print("No complex action is configured for mouse clicks.")
            return
        x = int(event.x // self.cell_size)
        y = int(event.y // self.cell_size)
        self.apply_action(self.keymap.mouse_action, {"x": x, "y": y})

    def apply_action(self, action: GameAction, data: dict[str, Any]) -> None:
        self.obs = self.env.step(action, data=data)
        self.step += 1
        self.draw()
        if self.obs and self.obs.state == GameState.WIN:
            print_status(self.step, self.obs)
            print("WIN")

    def draw(self) -> None:
        grid = frame_to_grid(self.obs.frame if self.obs else None)
        height = len(grid)
        width = len(grid[0]) if grid else 1
        self.canvas.config(width=width * self.cell_size, height=height * self.cell_size)
        self.canvas.delete("all")
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                left = x * self.cell_size
                top = y * self.cell_size
                self.canvas.create_rectangle(
                    left,
                    top,
                    left + self.cell_size,
                    top + self.cell_size,
                    fill=color_for_value(value),
                    outline="",
                )
        self.status.set(status_text(self.step, self.obs, self.keymap))


def load_keymap(path: Path, action_space: list[GameAction]) -> KeymapConfig:
    if not path.exists():
        raise SystemExit(f"Keymap file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file) if path.suffix.lower() == ".json" else yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Keymap file {path} must contain a mapping.")

    raw_keys = loaded.get("keys")
    if not isinstance(raw_keys, dict):
        raise SystemExit(f"Keymap file {path} must contain a 'keys' mapping.")

    configured_keys = {str(key): parse_action_name(str(value)) for key, value in raw_keys.items()}
    mouse_action = None
    if loaded.get("mouse_action") is not None:
        mouse_action = parse_action_name(str(loaded["mouse_action"]))

    available = set(action_space)
    keys = {
        key: action
        for key, action in configured_keys.items()
        if action == GameAction.RESET or (action in available and not action.is_complex())
    }

    complex_actions = [action for action in action_space if action.is_complex()]
    if mouse_action is None and len(complex_actions) == 1:
        mouse_action = complex_actions[0]
    if mouse_action is not None:
        if mouse_action not in available:
            if complex_actions:
                raise SystemExit(f"mouse_action {mouse_action.name} is not available.")
            mouse_action = None
        elif not mouse_action.is_complex():
            raise SystemExit(f"mouse_action {mouse_action.name} must be a complex action.")

    missing = sorted(action.name for action in action_space if not action.is_complex() and action not in set(keys.values()))
    if missing:
        raise SystemExit(f"Keymap is missing non-complex action(s): {', '.join(missing)}")
    return KeymapConfig(keys=keys, mouse_action=mouse_action)


def parse_action_name(raw: str) -> GameAction:
    token = raw.strip().upper()
    if token.isdigit():
        token = f"ACTION{token}" if token != "0" else "RESET"
    try:
        return GameAction[token]
    except KeyError as error:
        raise SystemExit(f"Unknown action {raw!r}.") from error


def color_for_value(value: int) -> str:
    return ARC_COLORS.get(value, f"#{(value * 37) % 256:02x}{(value * 67) % 256:02x}{(value * 97) % 256:02x}")


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


def print_help(action_space: list[GameAction], keymap: KeymapConfig) -> None:
    print("Keyboard controls:")
    for key, action in sorted(keymap.keys.items(), key=lambda item: item[0]):
        print(f"  {key:<10} {action.name}")
    if keymap.mouse_action:
        print(f"Mouse click -> {keymap.mouse_action.name} x y")
    print("  q/Escape   quit")
    print("Available actions:")
    for action in action_space:
        suffix = " x y" if action.is_complex() else ""
        print(f"  {action.name}{suffix}")


def print_status(step: int, obs: FrameDataRaw | None) -> None:
    print(status_text(step, obs, None))


def status_text(step: int, obs: FrameDataRaw | None, keymap: KeymapConfig | None) -> str:
    if obs is None:
        return f"step={step} state=None"
    mouse = f" mouse={keymap.mouse_action.name}" if keymap and keymap.mouse_action else ""
    return f"step={step} state={obs.state.name} levels_completed={obs.levels_completed}{mouse}"


if __name__ == "__main__":
    main()
