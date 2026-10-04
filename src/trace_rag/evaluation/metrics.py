"""Evaluation metrics for Person C experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set


@dataclass(frozen=True)
class RetrievalMetrics:
    precision: float
    recall: float
    f1: float

    def to_dict(self):
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
        }


def retrieval_metrics(
    retrieved: Iterable[str],
    relevant: Iterable[str],
) -> RetrievalMetrics:
    """Compute retrieval precision, recall, and F1."""
    retrieved_set: Set[str] = set(retrieved)
    relevant_set: Set[str] = set(relevant)

    if not retrieved_set and not relevant_set:
        return RetrievalMetrics(1.0, 1.0, 1.0)

    if not retrieved_set:
        return RetrievalMetrics(0.0, 0.0, 0.0)

    if not relevant_set:
        return RetrievalMetrics(0.0, 0.0, 0.0)

    true_positive = len(retrieved_set & relevant_set)

    precision = true_positive / len(retrieved_set)
    recall = true_positive / len(relevant_set)

    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
    )


def attack_success_rate(successes: int, attacks: int) -> float | None:
    """Fraction of attack attempts that succeed.

    Returns None when there are no attack cases.
    """
    if attacks <= 0:
        return None
    return successes / attacks


def false_positive_rate(
    false_positives: int,
    benign_cases: int,
) -> float | None:
    """Fraction of benign cases incorrectly flagged.

    Returns None when there are no benign cases.
    """
    if benign_cases <= 0:
        return None
    return false_positives / benign_cases


def detection_delay(
    attack_step: int,
    detection_step: int | None,
) -> int | None:
    """Number of stream steps between attack and first detection."""
    if detection_step is None:
        return None
    return max(0, detection_step - attack_step)


def exposure_window(
    attack_step: int,
    remediation_step: int | None,
    stream_end: int,
) -> int:
    """Number of steps for which an attack remains exposed."""
    end = stream_end if remediation_step is None else remediation_step
    return max(0, end - attack_step)


def remediation_recall(
    remediated: int,
    detected: int,
) -> float | None:
    """Fraction of detected attacks that were successfully remediated.

    Returns None when nothing was detected.
    """
    if detected <= 0:
        return None
    return remediated / detected


def llm_cost(llm_calls: int) -> int:
    """Return the number of LLM calls used by an evaluation case."""
    return max(0, int(llm_calls))
