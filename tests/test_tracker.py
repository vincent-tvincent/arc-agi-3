import numpy as np

from arc_agent.perception.object_tracker import ObjectTracker
from arc_agent.perception.object_types import ObjectComponent


def component(obj_id: int, color: int, x: int, y: int) -> ObjectComponent:
    return ObjectComponent(obj_id, color, np.array([[x, y]]), (x, y, x, y), (float(x), float(y)), 1, 0)


def test_stable_id_across_small_movement() -> None:
    tracker = ObjectTracker({"max_match_distance": 3})
    first = tracker.update([component(0, 1, 1, 1)]).objects[0].track_id
    second = tracker.update([component(0, 1, 2, 1)]).objects[0].track_id
    assert first == second


def test_detects_disappeared_object() -> None:
    tracker = ObjectTracker({"max_match_distance": 3})
    tracker.update([component(0, 1, 1, 1)])
    result = tracker.update([])
    assert len(result.disappeared_objects) == 1


def test_detects_new_object() -> None:
    tracker = ObjectTracker({"max_match_distance": 1})
    tracker.update([component(0, 1, 1, 1)])
    result = tracker.update([component(0, 1, 1, 1), component(1, 2, 5, 5)])
    assert len(result.new_objects) == 1
    assert result.new_objects[0].track_id is not None
