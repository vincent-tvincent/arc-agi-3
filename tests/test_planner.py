from arc_agent.execution.action_space import GridAction
from arc_agent.planning.astar import astar_path


def test_astar_reaches_target_around_obstacle() -> None:
    path = astar_path((0, 0), (4, 0), 5, 3, blocked={(1, 0), (2, 0), (3, 0)})
    assert path is not None
    assert len(path) == 6


def test_astar_avoids_high_risk_hazard_cells() -> None:
    path = astar_path((0, 0), (2, 0), 3, 3, blocked=set(), risk={(1, 0): 1.0})
    assert path is not None
    assert path[:2] != [int(GridAction.RIGHT), int(GridAction.RIGHT)]


def test_astar_returns_none_if_target_unreachable() -> None:
    path = astar_path((0, 0), (2, 2), 3, 3, blocked={(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)})
    assert path is None
