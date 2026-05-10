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
        recordings_dir: str = "recordings",
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
        self.env = None
        self.last_score = 0.0

    def reset(self, game_id: str | None = None, level_id: str | None = None) -> np.ndarray:
        del level_id
        self.game_id = game_id or self.game_id
        self.env = self._arcade.make(self.game_id, seed=self.seed, render_mode=self.render_mode, save_recording=True)
        if self.env is None:
            raise RuntimeError(f"Could not create ARC-AGI-3 environment {self.game_id!r}")
        obs = self.env.observation_space or self.env.step(self._game_action.RESET)
        self.last_score = float(getattr(obs, "score", 0.0) or 0.0)
        return np.asarray(obs.frame[0] if getattr(obs.frame, "ndim", 0) == 3 else obs.frame, dtype=np.int64)

    def step(self, action: int, data: dict[str, Any] | None = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.env is None:
            self.reset()
        action_enum = self._game_action[f"ACTION{action + 1}"] if action < 5 else self._game_action.ACTION6
        obs = self.env.step(action_enum, data=data or {})
        score = float(getattr(obs, "score", self.last_score) or 0.0)
        reward = score - self.last_score
        self.last_score = score
        state = getattr(obs, "state", None)
        state_name = getattr(state, "name", str(state))
        done = state_name in {"WIN", "GAME_OVER"}
        frame = np.asarray(obs.frame[0] if getattr(obs.frame, "ndim", 0) == 3 else obs.frame, dtype=np.int64)
        return frame, reward, done, {"state": state_name, "score": score}

    @property
    def action_space_n(self) -> int:
        if self.env is None:
            return 7
        return len(self.env.action_space)
