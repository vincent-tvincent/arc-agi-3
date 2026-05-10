"""Textual graph inspection helpers."""

from __future__ import annotations

from arc_agent.graph.graph_data import GraphState


def graph_to_text(graph: GraphState) -> str:
    lines = [f"frame={graph.frame_index} nodes={graph.num_nodes} edges={graph.edge_index.shape[1]}"]
    for idx, node in enumerate(graph.nodes):
        lines.append(
            f"node[{idx}] track={node.track_id} color={node.color_id} bbox={node.bbox} "
            f"centroid=({node.centroid[0]:.1f},{node.centroid[1]:.1f}) area={node.area}"
        )
    for edge_idx in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge_idx])
        target = int(graph.edge_index[1, edge_idx])
        values = ", ".join(
            f"{name}={float(graph.edge_attr[edge_idx, feat_idx]):.2f}"
            for feat_idx, name in enumerate(graph.edge_feature_names)
        )
        lines.append(f"edge[{edge_idx}] {source}->{target} {values}")
    return "\n".join(lines)
