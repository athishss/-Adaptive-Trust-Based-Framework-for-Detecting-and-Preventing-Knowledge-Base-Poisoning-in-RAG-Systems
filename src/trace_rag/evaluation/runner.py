"""Experiment matrix and result containers for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class EvaluationCase:
    attack: str
    baseline: str
    seed: int

    def to_dict(self):
        return {
            "attack": self.attack,
            "baseline": self.baseline,
            "seed": int(self.seed),
        }


@dataclass
class EvaluationResult:
    case: EvaluationCase
    attack_success_rate: Optional[float] = None
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positive_rate: Optional[float] = None
    detection_delay: Optional[int] = None
    exposure_window: int = 0
    remediation_recall: Optional[float] = None
    llm_calls: int = 0

    def to_dict(self):
        return {
            "case": self.case.to_dict(),
            "attack_success_rate": self.attack_success_rate,
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "false_positive_rate": self.false_positive_rate,
            "detection_delay": self.detection_delay,
            "exposure_window": int(self.exposure_window),
            "remediation_recall": self.remediation_recall,
            "llm_calls": int(self.llm_calls),
        }


def build_experiment_matrix(
    attacks: Sequence[str],
    baselines: Sequence[str],
    seeds: Sequence[int],
) -> List[EvaluationCase]:
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
            seed=int(seed),
        )
        for attack in attacks
        for baseline in baselines
        for seed in seeds
    ]


def run_experiment_matrix(
    cases: Sequence[EvaluationCase],
    execute_case: Callable[[EvaluationCase], EvaluationResult],
) -> List[EvaluationResult]:
    """Execute cases in deterministic order."""
    if not cases:
        raise ValueError("cases must not be empty")

    results: List[EvaluationResult] = []

    for case in cases:
        result = execute_case(case)

        if not isinstance(result, EvaluationResult):
            raise TypeError(
                "execute_case must return an EvaluationResult"
            )

        if result.case != case:
            raise ValueError(
                "execute_case returned a result for the wrong evaluation case"
            )

        results.append(result)

    return results


def run_experiments(
    attacks: Sequence[str],
    baselines: Sequence[str],
    seeds: Sequence[int],
    execute_case: Callable[[EvaluationCase], EvaluationResult],
) -> List[EvaluationResult]:
    """Build and execute the complete attack/baseline/seed matrix."""
    cases = build_experiment_matrix(
        attacks=attacks,
        baselines=baselines,
        seeds=seeds,
    )
    return run_experiment_matrix(cases, execute_case)
