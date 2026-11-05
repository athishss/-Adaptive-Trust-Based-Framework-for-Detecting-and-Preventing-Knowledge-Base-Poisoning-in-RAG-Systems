"""Experiment matrix and result containers for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .metrics import ANSWER_LEVEL_ASR_NOT_MEASURED


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
    """One attack/baseline/seed result.

    ``poison_exposure_rate`` and ``poison_citation_rate`` are context and
    citation measurements over *eligible attack attempts*.  They are not
    answer-success rates.  ``answer_level_asr`` stays ``None`` with status
    ``NOT MEASURED`` unless a harness supplies a predefined, attack-specific
    malicious-answer criterion.
    """

    case: EvaluationCase
    eligible_attack_attempts: int = 0
    poison_exposed_attempts: int = 0
    poison_exposure_rate: Optional[float] = None
    poison_cited_attempts: int = 0
    poison_citation_rate: Optional[float] = None
    answer_level_asr: Optional[float] = None
    answer_level_asr_status: str = ANSWER_LEVEL_ASR_NOT_MEASURED
    answer_level_asr_numerator: Optional[int] = None
    answer_level_asr_denominator: int = 0
    answer_level_asr_reason: Optional[str] = None
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positive_rate: Optional[float] = None
    # False when no document was flagged (precision) or no poison existed
    # (recall).  ``precision``/``recall`` above then hold 0.0 for backward
    # compatibility, and must be reported as n/a, not as a measured zero.
    precision_defined: bool = True
    recall_defined: bool = True
    withheld_poison_recall: Optional[float] = None
    withheld_clean_rate: Optional[float] = None
    detection_delay: Optional[int] = None
    exposure_window: int = 0
    remediation_recall: Optional[float] = None
    llm_calls: int = 0
    p99_latency_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "case": self.case.to_dict(),
            "eligible_attack_attempts": int(self.eligible_attack_attempts),
            "poison_exposed_attempts": int(self.poison_exposed_attempts),
            "poison_exposure_rate": self.poison_exposure_rate,
            "poison_cited_attempts": int(self.poison_cited_attempts),
            "poison_citation_rate": self.poison_citation_rate,
            "answer_level_asr": self.answer_level_asr,
            "answer_level_asr_status": self.answer_level_asr_status,
            "answer_level_asr_numerator": self.answer_level_asr_numerator,
            "answer_level_asr_denominator": int(self.answer_level_asr_denominator),
            "answer_level_asr_reason": self.answer_level_asr_reason,
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "false_positive_rate": self.false_positive_rate,
            "precision_defined": bool(self.precision_defined),
            "recall_defined": bool(self.recall_defined),
            "withheld_poison_recall": self.withheld_poison_recall,
            "withheld_clean_rate": self.withheld_clean_rate,
            "detection_delay": self.detection_delay,
            "exposure_window": int(self.exposure_window),
            "remediation_recall": self.remediation_recall,
            "llm_calls": int(self.llm_calls),
            "p99_latency_ms": self.p99_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "details": dict(self.details),
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
