"""Small deterministic grid environment for local validation and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arc_agent.execution.action_space import ACTION_DELTAS, GridAction


BACKGROUND = 0
AGENT = 1
WALL = 2
GOAL = 3
COLLECTIBLE = 4
PUSHABLE = 5
TRIGGER = 6
DOOR = 7
HAZARD = 8

COLOR_LABELS = {
    BACKGROUND: {"background"},
    AGENT: {"controllable"},
    WALL: {"blocking"},
    GOAL: {"goal_related"},
    COLLECTIBLE: {"collectible"},
    PUSHABLE: {"movable"},
    TRIGGER: {"trigger"},
    DOOR: {"blocking"},
    HAZARD: {"hazardous"},
}


@dataclass
class MockGridConfig:
    width: int = 9
    height: int = 9
    max_steps: int = 80
    seed: int = 42


class MockGridEnv:
    """Turn-based grid task with generic mechanics for pipeline smoke tests."""

    def __init__(self, config: dict[str, Any] | MockGridConfig | None = None) -> None:
        if isinstance(config, MockGridConfig):
            self.config = config
        else:
            data = config or {}
            self.config = MockGridConfig(
                width=int(data.get("width", 9)),
                height=int(data.get("height", 9)),
                max_steps=int(data.get("max_steps", 80)),
                seed=int(data.get("seed", 42)),
            )
        self.rng = np.random.default_rng(self.config.seed)
        self.grid = np.zeros((self.config.height, self.config.width), dtype=np.int64)
        self.agent_pos = (1, 1)
        self.steps = 0
        self.done = False
        self.door_open = False
        self.collected = False
        self.last_info: dict[str, Any] = {}

    @property
    def action_space_n(self) -> int:
        return 8

    def reset(self, game_id: str | None = None, level_id: str | None = None) -> np.ndarray:
        del game_id, level_id
        self.grid.fill(BACKGROUND)
        self.steps = 0
        self.done = False
        self.door_open = False
        self.collected = False
        self.last_info = {}
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL
        self.grid[2, 3:6] = WALL
        self.grid[4, 4] = HAZARD
        self.grid[3, 2] = COLLECTIBLE
        self.grid[5, 2] = PUSHABLE
        self.grid[6, 1] = TRIGGER
        self.grid[6, 4] = DOOR
        self.grid[self.config.height - 2, self.config.width - 2] = GOAL
        self.agent_pos = (1, 1)
        self.grid[self.agent_pos[1], self.agent_pos[0]] = AGENT
        return self.grid.copy()

    def step(self, action: int, data: dict[str, Any] | None = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        del data
        if self.done:
            return self.grid.copy(), 0.0, True, {"state": "DONE"}
        self.steps += 1
        reward = -0.01
        info: dict[str, Any] = {"blocked": False, "event": None}
        grid_action = GridAction(action) if action in set(item.value for item in GridAction) else GridAction.WAIT
        if grid_action in ACTION_DELTAS:
            reward += self._move(grid_action, info)
        elif grid_action == GridAction.INTERACT:
            reward += self._interact(info)
        elif grid_action == GridAction.WAIT:
            info["event"] = "wait"
        if self.steps >= self.config.max_steps:
            self.done = True
            info["state"] = "GAME_OVER"
        elif self.done:
            info["state"] = info.get("state", "WIN")
        else:
            info["state"] = "NOT_FINISHED"
        self.last_info = info
        return self.grid.copy(), reward, self.done, info

    def affordance_labels(self) -> dict[int, set[str]]:
        return {
            AGENT: {"controllable"},
            WALL: {"blocking"},
            DOOR: {"blocking"} if not self.door_open else {"trigger"},
            GOAL: {"goal_related"},
            COLLECTIBLE: {"collectible"},
            PUSHABLE: {"movable"},
            TRIGGER: {"trigger"},
            HAZARD: {"hazardous"},
        }

    def _move(self, action: GridAction, info: dict[str, Any]) -> float:
        dx, dy = ACTION_DELTAS[action]
        ax, ay = self.agent_pos
        target = (ax + dx, ay + dy)
        target_value = int(self.grid[target[1], target[0]])
        if target_value in {WALL, DOOR}:
            info.update({"blocked": True, "blocked_cell": target, "event": "blocked"})
            return -0.02
        if target_value == PUSHABLE:
            beyond = (target[0] + dx, target[1] + dy)
            beyond_value = int(self.grid[beyond[1], beyond[0]])
            if beyond_value != BACKGROUND:
                info.update({"blocked": True, "blocked_cell": target, "event": "push_blocked"})
                return -0.02
            self.grid[beyond[1], beyond[0]] = PUSHABLE
            info["event"] = "pushed"
        self.grid[ay, ax] = BACKGROUND
        self.agent_pos = target
        if target_value == COLLECTIBLE:
            self.collected = True
            info["event"] = "collected"
            reward = 0.25
        elif target_value == TRIGGER:
            self.door_open = True
            self.grid[6, 4] = BACKGROUND
            info["event"] = "triggered"
            reward = 0.15
        elif target_value == HAZARD:
            self.done = True
            info.update({"event": "hazard", "state": "GAME_OVER"})
            reward = -1.0
        elif target_value == GOAL:
            self.done = True
            info.update({"event": "goal", "state": "WIN"})
            reward = 1.0
        else:
            reward = 0.0
        if not self.done:
            self.grid[target[1], target[0]] = AGENT
        return reward

    def _interact(self, info: dict[str, Any]) -> float:
        ax, ay = self.agent_pos
        for dx, dy in ACTION_DELTAS.values():
            cell = (ax + dx, ay + dy)
            if int(self.grid[cell[1], cell[0]]) == TRIGGER:
                self.door_open = True
                self.grid[6, 4] = BACKGROUND
                info["event"] = "triggered"
                return 0.15
        info["event"] = "interact_empty"
        return -0.01
