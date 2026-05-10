#!/usr/bin/env python3
"""Inspect graph construction for one replay step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.graph.graph_visualizer import graph_to_text
from arc_agent.perception.object_tracker import ObjectTracker
from arc_agent.perception.segmenter import GridSegmenter
from arc_agent.training.replay_dataset import load_replay
from arc_agent.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--config", default="configs/debug_cpu.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    steps = load_replay(args.replay)
    row = steps[min(args.step, len(steps) - 1)]
    segmenter = GridSegmenter(config.get("perception", {}))
    tracker = ObjectTracker(config.get("tracking", {}))
    objects = tracker.update(segmenter.segment(row["frame"], frame_index=int(row["step"]))).objects
    graph = GraphBuilder(config.get("edge_builder", {}), config.get("features", {})).build(objects, frame_shape=(len(row["frame"]), len(row["frame"][0])))
    print(graph_to_text(graph))


if __name__ == "__main__":
    main()
