"""Stable environment adapter interface plus optional ARC-AGI-3 wrapper."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class EnvAdapter(Protocol):
    def reset(self, game_id: str | None = None, level_id: str | None = None) -> np.ndarray: ...

    def step(self, action: int, data: dict[str, Any] | None = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]: ...

    @property
    def action_space_n(self) -> int: ...


class ArcAgi3EnvAdapter:
    """Thin wrapper around the official toolkit when it is installed."""

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        offline: bool = True,
        environments_dir: str = "environment_files",
        recordings_dir: str = "/run/media/blue-lobster/disk3/CS274p_output/recordings",
        render_mode: str | None = None,
    ) -> None:
        try:
            import arc_agi
            from arc_agi import OperationMode
            from arcengine import GameAction
        except Exception as error:
            raise RuntimeError("Official ARC-AGI-3 toolkit is not installed in this environment.") from error
        mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
        self._game_action = GameAction
        self._arcade = arc_agi.Arcade(
            operation_mode=mode,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
        )
        self.game_id = game_id
        self.seed = seed
        self.render_mode = render_mode
        self.env: Any | None = None
        self.last_frame: np.ndarray | None = None
        self.last_score = 0.0

    def reset(self, game_id: str | None = None, level_id: str | None = None) -> np.ndarray:
        del level_id
        self.game_id = game_id or self.game_id
        self.env = self._arcade.make(self.game_id, seed=self.seed, render_mode=self.render_mode, save_recording=True)
        if self.env is None:
            raise RuntimeError(f"Could not create ARC-AGI-3 environment {self.game_id!r}")
        obs = self.env.observation_space or self.env.step(self._game_action.RESET)
        if obs is None:
            raise RuntimeError(f"ARC-AGI-3 environment {self.game_id!r} returned no reset observation.")
        self.last_score = float(getattr(obs, "score", 0.0) or 0.0)
        self.last_frame = self._frame_from_observation(obs)
        return self.last_frame

    def step(self, action: int, data: dict[str, Any] | None = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.env is None:
            self.reset()
        env = self.env
        if env is None:
            raise RuntimeError("ARC-AGI-3 environment is not initialized.")
        action_enum = self._action_enum_from_index(action)
        action_data = data if data is not None else self._default_action_data(action_enum)
        obs = env.step(action_enum, data=action_data)
        if obs is None:
            raise RuntimeError("ARC-AGI-3 environment returned no observation from step().")
        score = float(getattr(obs, "score", self.last_score) or 0.0)
        reward = score - self.last_score
        self.last_score = score
        state = getattr(obs, "state", None)
        state_name = getattr(state, "name", str(state))
        done = state_name in {"WIN", "GAME_OVER"}
        frame = self._frame_from_observation(obs)
        self.last_frame = frame
        return frame, reward, done, {"state": state_name, "score": score, "action": getattr(action_enum, "name", str(action_enum)), "action_data": action_data}

    @property
    def action_space_n(self) -> int:
        if self.env is None:
            return 7
        return len(self.env.action_space)

    def _action_enum_from_index(self, action: int) -> Any:
        if self.env is None:
            self.reset()
        available = list(getattr(self.env, "action_space", []) or [])
        action_name = "RESET" if action < 0 else f"ACTION{action + 1}"
        candidate = getattr(self._game_action, action_name, None)
        if candidate is not None and (not available or candidate in available):
            return candidate
        if 0 <= action < len(available):
            return available[action]
        if available:
            return available[0]
        return self._game_action.RESET

    def _frame_from_observation(self, obs: Any) -> np.ndarray:
        frame = getattr(obs, "frame", None)
        if frame is None:
            raise RuntimeError("ARC-AGI-3 observation has no frame.")
        return np.asarray(frame[0] if getattr(frame, "ndim", 0) == 3 else frame, dtype=np.int64)

    def _default_action_data(self, action_enum: Any) -> dict[str, int]:
        is_complex = getattr(action_enum, "is_complex", None)
        if not callable(is_complex) or not is_complex():
            return {}
        frame = self.last_frame
        if frame is None or frame.size == 0:
            return {"x": 0, "y": 0}
        values, counts = np.unique(frame, return_counts=True)
        background = int(values[int(np.argmax(counts))])
        points = np.argwhere(frame != background)
        if points.size == 0:
            y = int(frame.shape[0] // 2)
            x = int(frame.shape[1] // 2)
            return {"x": x, "y": y}
        y, x = points[len(points) // 2]
        return {"x": int(x), "y": int(y)}
