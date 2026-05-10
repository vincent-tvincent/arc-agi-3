from arc_agent.belief.belief_state import BeliefState
from arc_agent.belief.belief_update import BeliefUpdater, TransitionDelta


def test_failed_movement_increases_blocking_belief() -> None:
    belief = BeliefState()
    updater = BeliefUpdater({"alpha": 0.5})
    updater.update(belief, delta=TransitionDelta(failed_movement_blocker=7))
    assert belief.object_beliefs[7]["blocking"] > 0.1


def test_disappearance_increases_collectible_belief() -> None:
    belief = BeliefState()
    updater = BeliefUpdater({"alpha": 0.5})
    updater.update(belief, delta=TransitionDelta(disappeared_object_ids=[3]))
    assert belief.object_beliefs[3]["collectible"] > 0.1


def test_remote_change_creates_causal_relation() -> None:
    belief = BeliefState()
    updater = BeliefUpdater({"alpha": 0.5})
    updater.update(belief, delta=TransitionDelta(contact_object_id=1, remote_changed_object_ids=[2]))
    assert (1, 2) in belief.causal_edges
    assert belief.causal_edges[(1, 2)]["probability"] > 0.1
