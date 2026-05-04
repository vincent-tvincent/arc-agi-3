"""Analyze collected ARC-AGI-3 transitions and rank model-family hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3_pipeline.experience import Transition
from arc3_pipeline.model_families import generate_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", help="JSONL file produced by collect_experience.py")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    transitions = load_transitions(Path(args.run_file))
    hypotheses = generate_hypotheses(transitions)

    print(f"loaded {len(transitions)} transitions")
    for hypothesis in hypotheses[: args.top]:
        print()
        print(f"{hypothesis.family}: score={hypothesis.score:.2f} confidence={hypothesis.confidence}")
        for item in hypothesis.evidence:
            print(f"  - {item}")
        print(f"  parameters={json.dumps(hypothesis.parameters, sort_keys=True)}")


def load_transitions(path: Path) -> list[Transition]:
    transitions: list[Transition] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            transitions.append(Transition(**json.loads(line)))
    return transitions


if __name__ == "__main__":
    main()

