import numpy as np

from arc_agent.perception.segmenter import GridSegmenter


def test_connected_components_return_expected_count() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 2, 2],
        ]
    )
    objects = GridSegmenter({"min_component_area": 1}).segment(grid)
    assert len(objects) == 3


def test_background_ignored() -> None:
    grid = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    objects = GridSegmenter().segment(grid)
    assert len(objects) == 1
    assert objects[0].color_id == 5


def test_components_have_valid_bbox_and_centroid() -> None:
    grid = np.array([[0, 0, 0], [0, 2, 2], [0, 2, 0]])
    obj = GridSegmenter().segment(grid)[0]
    assert obj.bbox == (1, 1, 2, 2)
    assert obj.centroid == (4 / 3, 4 / 3)
    assert obj.area == 3
