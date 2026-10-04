"""Baseline policies for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from trace_rag.contracts import PolicyDecision, RetrievedDocument, VerificationResult
from trace_rag.trust.policy import TrustPolicy
from trace_rag.trust.verifier import CorroborationVerifier


@dataclass(frozen=True)
class BaselineResult:
    documents: List[RetrievedDocument]
    blocked: List[str]
    verified: List[VerificationResult]
    llm_calls: int = 0

    def to_dict(self):
        return {
            "documents": [
                d.to_dict() if hasattr(d, "to_dict") else d
                for d in self.documents
            ],
            "blocked": list(self.blocked),
            "verified": [
                v.to_dict() if hasattr(v, "to_dict") else v
                for v in self.verified
            ],
            "llm_calls": int(self.llm_calls),
        }


def no_defence(documents: Sequence[RetrievedDocument]) -> BaselineResult:
    return BaselineResult(
        documents=list(documents),
        blocked=[],
        verified=[],
        llm_calls=0,
    )


def duplicate_filter(documents: Sequence[RetrievedDocument]) -> BaselineResult:
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
    scorer,
    threshold: float,
) -> BaselineResult:
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
    kept = []
    blocked = []

    for doc in documents:
        if float(doc.trust.trust) < threshold:
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
    query,
    query_id,
    documents,
    verifier: CorroborationVerifier,
) -> BaselineResult:
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


class TrustRAGPolicy:
    """TrustRAG baseline backed by Person B's trust state machine."""

    def __init__(self, ledger, on_quarantine=None):
        self.policy = TrustPolicy(
            ledger=ledger,
            on_quarantine=on_quarantine,
        )

    def decide(
        self,
        query,
        query_id,
        documents,
        assessments,
        verifier=None,
        **kwargs,
    ) -> PolicyDecision:
        return self.policy.decide(
            query=query,
            query_id=query_id,
            documents=documents,
            assessments=assessments,
            verifier=verifier,
            **kwargs,
        )


class RobustRAGPolicy:
    """Stricter trust-aware baseline with corroboration gating.

    Documents whose effective retrieval trust is below ``trust_floor`` are
    excluded before normal TrustPolicy processing. Remaining MEDIUM-band
    documents still go through Person B corroboration and quarantine logic.
    """

    def __init__(self, ledger, trust_floor: float = 0.20, on_quarantine=None):
        self.policy = TrustPolicy(
            ledger=ledger,
            on_quarantine=on_quarantine,
        )
        self.trust_floor = float(trust_floor)

    def decide(
        self,
        query,
        query_id,
        documents,
        assessments,
        verifier=None,
        **kwargs,
    ) -> PolicyDecision:
        eligible = []
        pre_excluded = []

        for doc in documents:
            if float(doc.trust.trust) < self.trust_floor:
                pre_excluded.append(doc.doc_id)
            else:
                eligible.append(doc)

        eligible_ids = {doc.doc_id for doc in eligible}
        eligible_assessments = [
            assessment
            for assessment in assessments
            if assessment.doc_id in eligible_ids
        ]

        decision = self.policy.decide(
            query=query,
            query_id=query_id,
            documents=eligible,
            assessments=eligible_assessments,
            verifier=verifier,
            **kwargs,
        )

        excluded = tuple(pre_excluded) + tuple(decision.excluded_doc_ids)

        notes = dict(decision.notes)
        notes.update({
            "policy": "RobustRAGPolicy",
            "trust_floor": self.trust_floor,
            "pre_excluded_low_trust": list(pre_excluded),
        })

        return PolicyDecision(
            context_doc_ids=decision.context_doc_ids,
            excluded_doc_ids=excluded,
            verified=decision.verified,
            notes=notes,
        )


BASELINE_NAMES = (
    "no_defence",
    "perplexity",
    "duplicate_filter",
    "TrustRAG",
    "RobustRAG",
    "always_on_loo",
)


BASELINE_POLICIES = {
    "TrustRAG": TrustRAGPolicy,
    "RobustRAG": RobustRAGPolicy,
}
