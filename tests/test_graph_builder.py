import numpy as np

from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.perception.object_types import ObjectComponent


def component(obj_id: int, color: int, cells: list[tuple[int, int]]) -> ObjectComponent:
    pixels = np.array(cells)
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return ObjectComponent(
        id=obj_id,
        color_id=color,
        pixels=pixels,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
        area=len(cells),
        frame_index=0,
        track_id=obj_id + 1,
    )


def test_edge_shapes_and_touching_edge() -> None:
    graph = GraphBuilder({"spatial_radius": 3}, {"max_colors": 8, "action_space_n": 4}).build(
        [component(0, 1, [(1, 1)]), component(1, 2, [(2, 1)])],
        frame_shape=(4, 4),
    )
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == len(graph.edge_feature_names)
    assert any(float(row[3]) == 1.0 for row in graph.edge_attr)


def test_same_color_edge_exists_when_enabled() -> None:
    graph = GraphBuilder({"spatial_radius": 0, "include_same_color_edges": True}, {"max_colors": 8, "action_space_n": 4}).build(
        [component(0, 3, [(1, 1)]), component(1, 3, [(4, 4)])],
        frame_shape=(6, 6),
    )
    assert any(float(row[4]) == 1.0 for row in graph.edge_attr)


def test_rgb_like_frame_shape_uses_spatial_dimensions() -> None:
    graph = GraphBuilder({"spatial_radius": 3}, {"max_colors": 8, "action_space_n": 4}).build(
        [component(0, 1, [(1, 1)])],
        frame_shape=(10, 12, 3),
    )
    assert graph.graph_features[1] == 12
    assert graph.graph_features[2] == 10
