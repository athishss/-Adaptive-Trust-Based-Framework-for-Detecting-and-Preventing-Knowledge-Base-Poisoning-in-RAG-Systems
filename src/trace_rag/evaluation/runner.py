"""Experiment runner data structures for Person C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvaluationCase:
    """One attack/baseline/seed experiment configuration."""

    attack: str
    baseline: str
    seed: int


@dataclass
class EvaluationResult:
    """Metrics collected from one evaluation case."""

    case: EvaluationCase
    attack_success_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positive_rate: float = 0.0
    detection_delay: Optional[int] = None
    exposure_window: int = 0
    remediation_recall: float = 0.0
    llm_calls: int = 0

def build_experiment_matrix(
    attacks: list[str],
    baselines: list[str],
    seeds: list[int],
) -> list[EvaluationCase]:
    """Create the attack x baseline x seed experiment matrix."""
    if not attacks:
        raise ValueError("attacks must not be empty")
    if not baselines:
        raise ValueError("baselines must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")

    return [
        EvaluationCase(
            attack=attack,
            baseline=baseline,
            seed=seed,
        )
        for attack in attacks
        for baseline in baselines
        for seed in seeds
    ]
