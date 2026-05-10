"""Small file IO helpers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with Path(path).open(mode, encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def write_json_gz(path: str | Path, data: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(data, file)


def read_json_gz(path: str | Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)
