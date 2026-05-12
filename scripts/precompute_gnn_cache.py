#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`arc_agent.cli.precompute_gnn_cache`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from arc_agent.cli.precompute_gnn_cache import main


if __name__ == "__main__":
    main()
