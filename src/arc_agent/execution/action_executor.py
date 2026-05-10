"""Convert selected subgoals into a single executable low-level action."""

from __future__ import annotations

from arc_agent.execution.action_space import GridAction
from arc_agent.planning.candidates import CandidateSubgoal


class ActionExecutor:
    def next_action(self, candidate: CandidateSubgoal | None) -> int:
        if candidate is None or not candidate.planner_path:
            return int(GridAction.WAIT)
        return int(candidate.planner_path[0])
