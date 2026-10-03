"""Baseline policies for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from trace_rag.contracts import RetrievedDocument, VerificationResult
from trace_rag.trust.verifier import CorroborationVerifier


@dataclass(frozen=True)
class BaselineResult:
    """Result produced by a baseline policy."""

    documents: List[RetrievedDocument]
    blocked: List[str]
    verified: List[VerificationResult]
    llm_calls: int = 0


def no_defence(documents: Sequence[RetrievedDocument]) -> BaselineResult:
    """Return all retrieved documents without security filtering."""
    return BaselineResult(
        documents=list(documents),
        blocked=[],
        verified=[],
        llm_calls=0,
    )


def duplicate_filter(
    documents: Sequence[RetrievedDocument],
) -> BaselineResult:
    """Remove exact duplicate passages."""
    seen = set()
    kept = []
    blocked = []

    for doc in documents:
        key = doc.text.strip().lower()

        if key in seen:
            blocked.append(doc.doc_id)
        else:
            seen.add(key)
            kept.append(doc)

    return BaselineResult(
        documents=kept,
        blocked=blocked,
        verified=[],
        llm_calls=0,
    )


def perplexity_filter(
    documents: Sequence[RetrievedDocument],
    scorer: Callable[[str], float],
    threshold: float,
) -> BaselineResult:
    """Filter passages whose perplexity exceeds the threshold."""
    kept = []
    blocked = []

    for doc in documents:
        if float(scorer(doc.text)) > threshold:
            blocked.append(doc.doc_id)
        else:
            kept.append(doc)

    return BaselineResult(
        documents=kept,
        blocked=blocked,
        verified=[],
        llm_calls=0,
    )


def trust_threshold_filter(
    documents: Sequence[RetrievedDocument],
    threshold: float,
) -> BaselineResult:
    """Keep passages whose existing trust score meets the threshold."""
    kept = []
    blocked = []

    for doc in documents:
        if float(doc.trust) < threshold:
            blocked.append(doc.doc_id)
        else:
            kept.append(doc)

    return BaselineResult(
        documents=kept,
        blocked=blocked,
        verified=[],
        llm_calls=0,
    )


def always_on_loo(
    query: str,
    query_id: str,
    documents: Sequence[RetrievedDocument],
    verifier: CorroborationVerifier,
) -> BaselineResult:
    """Verify every retrieved passage against the remaining pool.

    This is the always-on verification baseline. Unlike the adaptive
    TrustPolicy, it does not first require a MEDIUM suspicion band.
    """
    docs = list(documents)
    kept = []
    blocked = []
    verified = []
    llm_calls = 0

    for target in docs:
        result = verifier.verify(
            query=query,
            query_id=query_id,
            target=target,
            pool=docs,
        )

        verified.append(result)
        llm_calls += int(result.llm_calls)

        if result.outcome.value == "REFUTE":
            blocked.append(target.doc_id)
        else:
            kept.append(target)

    return BaselineResult(
        documents=kept,
        blocked=blocked,
        verified=verified,
        llm_calls=llm_calls,
    )


BASELINE_NAMES = (
    "no_defence",
    "perplexity",
    "duplicate_filter",
    "TrustRAG",
    "RobustRAG",
    "always_on_loo",
)


__all__ = [
    "BaselineResult",
    "no_defence",
    "duplicate_filter",
    "perplexity_filter",
    "trust_threshold_filter",
    "always_on_loo",
    "BASELINE_NAMES",
]
