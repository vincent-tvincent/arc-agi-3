#!/usr/bin/env python3
"""Print a saved replay in compact textual form."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from arc_agent.training.replay_dataset import load_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    args = parser.parse_args()
    steps = load_replay(args.replay)
    for row in steps:
        print(f"step={row.get('step')} action={row.get('action')} reward={row.get('reward')} done={row.get('done')} info={row.get('info')}")


if __name__ == "__main__":
    main()
