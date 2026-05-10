#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`arc_agent.cli.evaluate_agent`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from arc_agent.cli.evaluate_agent import main


if __name__ == "__main__":
    main()
