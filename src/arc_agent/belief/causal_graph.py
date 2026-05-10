"""Small helper around probabilistic causal object relations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalGraph:
    edges: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)

    def increase(self, source: int, target: int, amount: float, evidence: str = "transition") -> float:
        data = self.edges.setdefault((source, target), {"probability": 0.1, "evidence_count": 0.0, "last_evidence": ""})
        old = float(data.get("probability", 0.1))
        new = min(0.99, old + amount * (1.0 - old))
        data["probability"] = new
        data["evidence_count"] = float(data.get("evidence_count", 0.0)) + 1.0
        data["last_evidence"] = evidence
        return new
