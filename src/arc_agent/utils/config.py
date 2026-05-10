"""YAML configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML file and apply optional ``key.path=value`` overrides."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    for override in overrides or []:
        apply_override(data, override)
    return data


def apply_override(config: MutableMapping[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise ValueError(f"Override must look like key.path=value, got {raw!r}")
    key, raw_value = raw.split("=", 1)
    value = yaml.safe_load(raw_value)
    cursor: MutableMapping[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot assign through non-mapping config key {part!r}")
        cursor = child
    cursor[parts[-1]] = value


def deep_merge(base: Config, update: Mapping[str, Any]) -> Config:
    merged: Config = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def get_config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def run_dir(config: Mapping[str, Any]) -> Path:
    output_dir = Path(str(get_config_value(config, "project.output_dir", "/run/media/blue-lobster/disk3/CS274p_output/runs")))
    run_name = str(get_config_value(config, "project.run_name", "run"))
    return output_dir / run_name
