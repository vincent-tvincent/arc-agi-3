"""Metric helpers for evaluation and training loops."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def success_rate(successes: list[bool]) -> float:
    return sum(1 for item in successes if item) / len(successes) if successes else 0.0


def binary_f1(predictions: list[int], targets: list[int]) -> float:
    tp = sum(1 for pred, target in zip(predictions, targets, strict=False) if pred == 1 and target == 1)
    fp = sum(1 for pred, target in zip(predictions, targets, strict=False) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, targets, strict=False) if pred == 0 and target == 1)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0
