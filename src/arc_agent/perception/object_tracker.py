"""Object tracking through Hungarian or nearest-neighbor assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

import numpy as np

from arc_agent.perception.object_types import ObjectComponent


@dataclass
class TrackingResult:
    objects: list[ObjectComponent]
    new_objects: list[ObjectComponent] = field(default_factory=list)
    disappeared_objects: list[ObjectComponent] = field(default_factory=list)
    moved_objects: list[ObjectComponent] = field(default_factory=list)


class ObjectTracker:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.max_match_distance = float(config.get("max_match_distance", 4))
        self.iou_weight = float(config.get("iou_weight", 0.4))
        self.color_weight = float(config.get("color_weight", 0.3))
        self.position_weight = float(config.get("position_weight", 0.3))
        self.next_track_id = 1
        self.previous: list[ObjectComponent] = []
        self.last_result = TrackingResult(objects=[])

    def reset(self) -> None:
        self.next_track_id = 1
        self.previous = []
        self.last_result = TrackingResult(objects=[])

    def update(self, objects: list[ObjectComponent]) -> TrackingResult:
        if not self.previous:
            for obj in objects:
                obj.track_id = self._new_track_id()
                obj.attributes["is_new"] = True
            result = TrackingResult(objects=objects, new_objects=list(objects))
            self.previous = [self._copy_component(obj) for obj in objects]
            self.last_result = result
            return result

        assignments = self._assign(self.previous, objects)
        matched_previous: set[int] = set()
        matched_current: set[int] = set()
        moved: list[ObjectComponent] = []

        for prev_idx, cur_idx in assignments:
            prev = self.previous[prev_idx]
            current = objects[cur_idx]
            current.track_id = prev.track_id
            current.attributes["previous_centroid"] = prev.centroid
            current.attributes["dx"] = current.centroid[0] - prev.centroid[0]
            current.attributes["dy"] = current.centroid[1] - prev.centroid[1]
            current.attributes["is_new"] = False
            current.attributes["changed"] = bool(current.attributes["dx"] or current.attributes["dy"] or current.area != prev.area)
            if current.attributes["changed"]:
                moved.append(current)
            matched_previous.add(prev_idx)
            matched_current.add(cur_idx)

        new_objects: list[ObjectComponent] = []
        for idx, obj in enumerate(objects):
            if idx not in matched_current:
                obj.track_id = self._new_track_id()
                obj.attributes["is_new"] = True
                obj.attributes["dx"] = 0.0
                obj.attributes["dy"] = 0.0
                new_objects.append(obj)

        disappeared = [self.previous[idx] for idx in range(len(self.previous)) if idx not in matched_previous]
        for obj in disappeared:
            obj.attributes["disappeared"] = True

        result = TrackingResult(objects=objects, new_objects=new_objects, disappeared_objects=disappeared, moved_objects=moved)
        self.previous = [self._copy_component(obj) for obj in objects]
        self.last_result = result
        return result

    def _assign(self, previous: list[ObjectComponent], current: list[ObjectComponent]) -> list[tuple[int, int]]:
        if not previous or not current:
            return []
        cost = np.zeros((len(previous), len(current)), dtype=np.float64)
        for i, prev in enumerate(previous):
            for j, cur in enumerate(current):
                cost[i, j] = self._cost(prev, cur)
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(cost)
            pairs = list(zip(rows.tolist(), cols.tolist(), strict=False))
        except Exception:
            pairs = self._nearest_pairs(cost)
        return [(i, j) for i, j in pairs if cost[i, j] <= 1.0 and self._distance(previous[i], current[j]) <= self.max_match_distance]

    def _nearest_pairs(self, cost: np.ndarray) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        candidates = sorted((float(cost[i, j]), i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1]))
        for _, row, col in candidates:
            if row not in used_rows and col not in used_cols:
                pairs.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        return pairs

    def _cost(self, prev: ObjectComponent, cur: ObjectComponent) -> float:
        distance = self._distance(prev, cur)
        normalized_distance = min(distance / max(self.max_match_distance, 1e-6), 1.0)
        color_mismatch = 0.0 if prev.color_id == cur.color_id else 1.0
        iou_cost = 1.0 - bbox_iou(prev.bbox, cur.bbox)
        return (
            self.position_weight * normalized_distance
            + self.color_weight * color_mismatch
            + self.iou_weight * iou_cost
        )

    def _distance(self, prev: ObjectComponent, cur: ObjectComponent) -> float:
        return hypot(prev.centroid[0] - cur.centroid[0], prev.centroid[1] - cur.centroid[1])

    def _new_track_id(self) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _copy_component(self, obj: ObjectComponent) -> ObjectComponent:
        return ObjectComponent(
            id=obj.id,
            color_id=obj.color_id,
            pixels=obj.pixels.copy(),
            bbox=obj.bbox,
            centroid=obj.centroid,
            area=obj.area,
            frame_index=obj.frame_index,
            track_id=obj.track_id,
            attributes=dict(obj.attributes),
        )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return inter / float(area_a + area_b - inter)
