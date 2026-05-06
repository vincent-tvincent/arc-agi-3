"""Analyze collected ARC-AGI-3 transitions and rank model-family hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from arc3_pipeline.experience import Transition
from arc3_pipeline.model_families import Hypothesis, generate_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", help="JSONL file produced by collect_experience.py")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json-out", help="Write machine-readable analysis JSON to this path.")
    parser.add_argument("--csv-out", help="Write human-debug hypothesis CSV to this path.")
    args = parser.parse_args()

    run_file = Path(args.run_file)
    transitions = load_transitions(run_file)
    hypotheses = generate_hypotheses(transitions)
    analysis = build_analysis(run_file, transitions, hypotheses)

    print(f"loaded {len(transitions)} transitions")
    for hypothesis in hypotheses[: args.top]:
        print()
        print(f"{hypothesis.family}: score={hypothesis.score:.2f} confidence={hypothesis.confidence}")
        for item in hypothesis.evidence:
            print(f"  - {item}")
        print(f"  parameters={json.dumps(hypothesis.parameters, sort_keys=True)}")

    json_out = Path(args.json_out) if args.json_out else default_json_path(run_file)
    csv_out = Path(args.csv_out) if args.csv_out else default_csv_path(run_file)

    write_json(json_out, analysis)
    print(f"\nwrote {json_out}")

    write_csv(csv_out, analysis)
    print(f"wrote {csv_out}")


def load_transitions(path: Path) -> list[Transition]:
    transitions: list[Transition] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            transitions.append(Transition(**json.loads(line)))
    return transitions


def build_analysis(run_file: Path, transitions: list[Transition], hypotheses: list[Hypothesis]) -> dict[str, Any]:
    top = hypotheses[0] if hypotheses else None
    second = hypotheses[1] if len(hypotheses) > 1 else None
    game_ids = sorted({transition.game_id for transition in transitions})
    seed = infer_seed(run_file)

    return {
        "analysis_id": run_file.stem,
        "run_file": str(run_file),
        "game_id": game_ids[0] if len(game_ids) == 1 else None,
        "game_ids": game_ids,
        "seed": seed,
        "transition_count": len(transitions),
        "top_family": top.family if top else None,
        "key_actions": extract_actions(top) if top else [],
        "second_family": second.family if second else None,
        "second_actions": extract_actions(second) if second else [],
        "hypotheses": [
            {
                "rank": rank,
                "key_actions": extract_actions(hypothesis),
                **hypothesis.to_json(),
            }
            for rank, hypothesis in enumerate(hypotheses, start=1)
        ],
    }


def extract_actions(hypothesis: Hypothesis | None) -> list[str]:
    if hypothesis is None:
        return []

    parameters = hypothesis.parameters
    for key in ("coordinate_actions", "candidate_actions", "progress_actions"):
        actions = parameters.get(key)
        if actions:
            return sorted(str(action) for action in actions)

    action_map = parameters.get("action_map")
    if isinstance(action_map, dict) and action_map:
        return sorted(str(action) for action in action_map)

    return []


def infer_seed(run_file: Path) -> int | None:
    marker = "_seed"
    stem = run_file.stem
    if marker not in stem:
        return None
    suffix = stem.rsplit(marker, 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None


def default_json_path(run_file: Path) -> Path:
    return Path("analysis") / f"{run_file.stem}.analysis.json"


def default_csv_path(run_file: Path) -> Path:
    return Path("analysis_csv") / f"{run_file.stem}.analysis.csv"


def write_json(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2, sort_keys=True)
        file.write("\n")


def write_csv(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "analysis_id",
        "run_file",
        "game_id",
        "seed",
        "transition_count",
        "rank",
        "family",
        "score",
        "confidence",
        "key_actions",
        "evidence",
        "parameters",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hypothesis in analysis["hypotheses"]:
            writer.writerow(
                {
                    "analysis_id": analysis["analysis_id"],
                    "run_file": analysis["run_file"],
                    "game_id": analysis["game_id"] or ",".join(analysis["game_ids"]),
                    "seed": analysis["seed"],
                    "transition_count": analysis["transition_count"],
                    "rank": hypothesis["rank"],
                    "family": hypothesis["family"],
                    "score": f"{hypothesis['score']:.2f}",
                    "confidence": hypothesis["confidence"],
                    "key_actions": ",".join(hypothesis["key_actions"]),
                    "evidence": " | ".join(hypothesis["evidence"]),
                    "parameters": json.dumps(hypothesis["parameters"], sort_keys=True),
                }
            )


if __name__ == "__main__":
    main()
