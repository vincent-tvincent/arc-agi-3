#!/usr/bin/env python3
"""Replay JSON/JSONL ARC action records through an official game and export a GIF."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np


try:
    from arc_agi.rendering import COLOR_MAP as OFFICIAL_COLOR_MAP
except Exception:
    OFFICIAL_COLOR_MAP = {
        0: "#FFFFFFFF",
        1: "#CCCCCCFF",
        2: "#999999FF",
        3: "#666666FF",
        4: "#333333FF",
        5: "#000000FF",
        6: "#E53AA3FF",
        7: "#FF7BCCFF",
        8: "#F93C31FF",
        9: "#1E93FFFF",
        10: "#88D8F1FF",
        11: "#FFDC00FF",
        12: "#FF851BFF",
        13: "#921231FF",
        14: "#4FCC30FF",
        15: "#A356D6FF",
    }


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


PALETTE = np.asarray([_hex_to_rgb(OFFICIAL_COLOR_MAP[idx]) for idx in range(16)], dtype=np.uint8)


@dataclass(frozen=True)
class ReplayAction:
    name: str
    data: dict[str, Any]
    source_index: int
    source_state: str | None = None

    def label(self) -> str:
        if self.data:
            data = ", ".join(f"{key}={value}" for key, value in sorted(self.data.items()))
            return f"{self.name}({data})"
        return self.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume ARC JSON/JSONL records, replay the actions in the official game, and export an annotated GIF."
    )
    parser.add_argument("record_path", help="Path to a JSON object/list or JSONL file containing action records.")
    parser.add_argument("--game", default=None, help="Game id or prefix, e.g. ls20 or ls20-9607627b. If omitted, infer from record/file name.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the official game replay.")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit the number of actions replayed.")
    parser.add_argument("--out", default=None, help="Output GIF path. Defaults to src/playground/replays/<input-stem>_<game>_seedN.gif.")
    parser.add_argument("--scale", type=int, default=8, help="Pixels per ARC cell in the GIF.")
    parser.add_argument("--duration-ms", type=int, default=450, help="Frame duration in milliseconds.")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True, help="Use already-downloaded games by default.")
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--recordings-dir", default="/run/media/blue-lobster/disk3/CS274p_output/recordings")
    args = parser.parse_args()

    record_path = Path(args.record_path)
    records = load_records(record_path)
    if not records:
        raise SystemExit(f"No JSON records found in {record_path}")

    arc, game_action = make_arcade(args.offline, args.environments_dir, args.recordings_dir)
    game_id = resolve_game_id(args.game, records, record_path, arc)
    actions = extract_actions(records, game_action)
    if args.max_steps is not None:
        actions = actions[: max(args.max_steps, 0)]
    if not actions:
        raise SystemExit(f"No replayable actions found in {record_path}")

    out_path = Path(args.out) if args.out else default_out_path(record_path, game_id, args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames, metadata = replay_to_frames(
        arc=arc,
        game_action=game_action,
        game_id=game_id,
        seed=args.seed,
        actions=actions,
        scale=max(1, args.scale),
    )
    save_gif(frames, out_path, max(1, args.duration_ms))
    metadata.update(
        {
            "input": str(record_path),
            "output_gif": str(out_path),
            "seed": args.seed,
            "offline": args.offline,
            "note": "Replay starts from a fresh official environment. Match the original seed to reproduce seed-dependent records.",
        }
    )
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"exported replay GIF to {out_path}")
    print(f"wrote metadata to {metadata_path}")
    print(f"game={game_id} seed={args.seed} actions={len(actions)} frames={len(frames)}")


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"Line {line_no} is JSON but not an object.")
            records.append(row)
        return records
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(row, dict) for row in parsed):
            raise SystemExit("JSON list must contain only objects.")
        return parsed
    raise SystemExit("Input JSON must be an object, a list of objects, or JSONL objects.")


def make_arcade(offline: bool, environments_dir: str, recordings_dir: str) -> tuple[Any, Any]:
    try:
        import arc_agi
        from arc_agi import OperationMode
        from arcengine import GameAction
    except Exception as error:
        raise SystemExit("Official ARC-AGI-3 toolkit is required. Run with the project ARC venv, e.g. ../arc_agi/bin/python.") from error
    mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
    return (
        arc_agi.Arcade(operation_mode=mode, environments_dir=environments_dir, recordings_dir=recordings_dir),
        GameAction,
    )


def resolve_game_id(game_arg: str | None, records: list[dict[str, Any]], path: Path, arc: Any) -> str:
    available = sorted(env.game_id for env in arc.get_environments())
    candidates: list[str] = []
    if game_arg:
        candidates.append(game_arg)
    for record in records:
        candidates.extend(record_game_candidates(record))
    candidates.extend(path_game_candidates(path.name))

    for candidate in candidates:
        matches = [game_id for game_id in available if game_id == candidate or game_id.startswith(candidate)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            shown = ", ".join(matches[:20])
            raise SystemExit(f"Game prefix {candidate!r} is ambiguous. Matches: {shown}")

    shown = ", ".join(available[:50])
    source = game_arg or path.name
    raise SystemExit(f"Could not infer game id from {source!r}. Use --game. Available games include: {shown}")


def record_game_candidates(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    data = record.get("data")
    if isinstance(data, dict):
        for key in ("game_id", "game"):
            value = data.get(key)
            if isinstance(value, str):
                candidates.append(value)
        action_input = data.get("action_input")
        if isinstance(action_input, dict):
            action_data = action_input.get("data")
            if isinstance(action_data, dict) and isinstance(action_data.get("game_id"), str):
                candidates.append(action_data["game_id"])
    for key in ("game_id", "game"):
        value = record.get(key)
        if isinstance(value, str):
            candidates.append(value)
    return candidates


def path_game_candidates(name: str) -> list[str]:
    candidates: list[str] = []
    match = re.search(r"([a-z][a-z0-9]{3})(?:[-_][0-9a-f]{8})?", name, flags=re.IGNORECASE)
    if match:
        candidates.append(match.group(0).replace("_", "-"))
        candidates.append(match.group(1))
    return candidates


def extract_actions(records: list[dict[str, Any]], game_action: Any) -> list[ReplayAction]:
    actions: list[ReplayAction] = []
    for index, record in enumerate(records):
        action_name, action_data, state = action_from_record(record, game_action)
        if action_name is None:
            continue
        actions.append(ReplayAction(action_name, action_data, index, state))
    return actions


def action_from_record(record: dict[str, Any], game_action: Any) -> tuple[str | None, dict[str, Any], str | None]:
    data = record.get("data")
    if isinstance(data, dict) and isinstance(data.get("action_input"), dict):
        action_input = data["action_input"]
        return action_name_from_raw(action_input.get("id"), game_action), clean_action_data(action_input.get("data")), data.get("state")

    if "action" in record:
        action_data = record.get("action_data", record.get("data", {}))
        return action_name_from_raw(record.get("action"), game_action), clean_action_data(action_data), record.get("state")

    nested_input = record.get("input")
    if isinstance(nested_input, dict) and "action" in record:
        return action_name_from_raw(record.get("action"), game_action), clean_action_data(record.get("action_data", {})), record.get("state")
    return None, {}, None


def action_name_from_raw(raw: Any, game_action: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return "RESET" if raw == 0 else f"ACTION{raw}"
    if isinstance(raw, str):
        token = raw.strip().upper()
        if not token:
            return None
        if token.isdigit():
            value = int(token)
            return "RESET" if value == 0 else f"ACTION{value}"
        if token.startswith("GAMEACTION."):
            token = token.split(".", 1)[1]
        return token
    name = getattr(raw, "name", None)
    if isinstance(name, str):
        return name
    return None


def clean_action_data(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key in ("x", "y"):
        if key in raw:
            try:
                cleaned[key] = int(raw[key])
            except (TypeError, ValueError):
                cleaned[key] = raw[key]
    return cleaned


def replay_to_frames(
    arc: Any,
    game_action: Any,
    game_id: str,
    seed: int,
    actions: list[ReplayAction],
    scale: int,
) -> tuple[list[Any], dict[str, Any]]:
    env = arc.make(game_id, seed=seed, render_mode=None, save_recording=False)
    if env is None:
        raise SystemExit(f"Could not create official ARC environment {game_id!r}.")
    obs = env.observation_space or env.step(game_action.RESET)
    if obs is None:
        raise SystemExit(f"Environment {game_id!r} returned no initial observation.")

    frames = [annotated_frame(obs, f"{game_id} seed={seed} | initial/reset", scale)]
    operations: list[dict[str, Any]] = []
    for step, replay_action in enumerate(actions, start=1):
        action_enum = game_action_from_name(game_action, replay_action.name, env.action_space)
        data = replay_action.data if action_is_complex(action_enum) else {}
        try:
            obs = env.step(action_enum, data=data)
        except Exception as error:
            raise SystemExit(f"Replay failed at step {step} using {replay_action.label()}: {error}") from error
        if obs is None:
            operations.append({"step": step, "record_index": replay_action.source_index, "action": replay_action.label(), "result": "no observation"})
            continue
        state_name = getattr(getattr(obs, "state", None), "name", str(getattr(obs, "state", None)))
        label = f"step {step:03d} | record {replay_action.source_index} | {replay_action.label()} | state={state_name}"
        frames.append(annotated_frame(obs, label, scale))
        operations.append(
            {
                "step": step,
                "record_index": replay_action.source_index,
                "action": replay_action.label(),
                "state": state_name,
                "source_state": replay_action.source_state,
            }
        )
    return frames, {"game_id": game_id, "actions": operations}


def game_action_from_name(game_action: Any, action_name: str, action_space: list[Any]) -> Any:
    candidate = getattr(game_action, action_name, None)
    if candidate is None:
        available_names = [getattr(action, "name", str(action)) for action in action_space]
        raise SystemExit(f"Unknown action {action_name!r}. Available in enum/action space: {available_names}")
    return candidate


def action_is_complex(action_enum: Any) -> bool:
    is_complex = getattr(action_enum, "is_complex", None)
    return bool(callable(is_complex) and is_complex())


def annotated_frame(obs: Any, text: str, scale: int) -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as error:
        raise SystemExit("Pillow is required to export GIFs. Install Pillow or use the ../arc_agi venv.") from error

    grid = frame_to_grid(getattr(obs, "frame", None))
    image = Image.fromarray(render_grid(grid, scale), mode="RGB")
    font = ImageFont.load_default()
    text_height = 18
    canvas = Image.new("RGB", (image.width, image.height + text_height), (20, 20, 20))
    canvas.paste(image, (0, text_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return canvas


def frame_to_grid(frame: Any) -> np.ndarray:
    if frame is None:
        raise SystemExit("Observation has no frame.")
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise SystemExit(f"Expected a 2D ARC frame, got shape {array.shape}.")
    return array.astype(np.int64, copy=False)


def render_grid(grid: np.ndarray, scale: int) -> np.ndarray:
    clipped = np.clip(grid, 0, len(PALETTE) - 1)
    rgb = PALETTE[clipped]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb.astype(np.uint8, copy=False)


def save_gif(frames: list[Any], out_path: Path, duration_ms: int) -> None:
    if not frames:
        raise SystemExit("No frames were produced.")
    first, *rest = frames
    first.save(out_path, save_all=True, append_images=rest, duration=duration_ms, loop=0)


def default_out_path(record_path: Path, game_id: str, seed: int) -> Path:
    safe_game = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in game_id)
    return Path("src") / "playground" / "replays" / f"{record_path.stem}_{safe_game}_seed{seed}.gif"


if __name__ == "__main__":
    main()
