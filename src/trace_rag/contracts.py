"""Shared data contracts and integration protocols for TRACE-RAG.

This module is the *only* thing Person B (trust / verification / policy) and
Person C (attacks / baselines / evaluation) need to import in order to plug
into Person A's pipeline.  Nothing here imports heavy optional dependencies,
so it is safe to import from any environment.

Design rules (do not break these, integration depends on them):
  * every payload object is a frozen dataclass with ``to_dict()`` producing
    JSON-serialisable primitives only;
  * every field name matches the plan document, Section 8 ("Contracts");
  * protocols are ``runtime_checkable`` so tests can assert conformance;
  * neutral/null implementations are provided so Person A's pipeline runs
    standalone before B's and C's components exist.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "Band", "TrustStatus", "VerificationOutcome", "Action",
    "TrustSnapshot", "RetrievedDocument", "SignalVector", "FeatureSnapshot",
    "SecurityAssessment", "VerificationResult", "PolicyDecision", "Citation",
    "AnswerRecord", "LLMResponse",
    "Embedder", "VectorIndex", "TrustProvider", "Verifier", "Policy", "LLM",
    "NullTrustProvider", "NullVerifier", "DefaultPolicy",
    "SIGNAL_NAMES", "FEATURE_NAMES", "CONTRACT_VERSION",
]

CONTRACT_VERSION = "1.0.0"


class Band(str, Enum):
    """Escalation band produced by the suspicion scorer (Person A)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TrustStatus(str, Enum):
    """Lifecycle state of a document in the trust ledger (Person B owns writes)."""

    TRUSTED = "TRUSTED"
    MONITORED = "MONITORED"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


class VerificationOutcome(str, Enum):
    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    NEUTRAL = "NEUTRAL"


class Action(str, Enum):
    """What the pipeline did with a retrieved passage for one query."""

    USE = "USE"                    # LOW band, used as context
    VERIFY_THEN_USE = "VERIFY"     # MEDIUM band, verified before answering
    EXCLUDE = "EXCLUDE"            # HIGH band, dropped from this answer
    BLOCKED = "BLOCKED"            # quarantined/rejected, never retrieved


# Order matters: it defines the column order of the scorer's design matrix.
SIGNAL_NAMES: Tuple[str, ...] = (
    "s1_query_echo",
    "s2_similarity_outlier",
    "s3_cluster_tightness",
    "s4_ingestion_burst",
    "s5_source_immaturity",
    "s6_neighbourhood_density",
)

# Raw sub-features exposed alongside the six signals.  Keeping them separate
# lets the scorer use them while ablations can switch them off.
EXTRA_FEATURE_NAMES: Tuple[str, ...] = (
    "x_similarity",
    "x_rank",
    "x_source_log_age_days",
    "x_source_n_docs",
    "x_source_trust",
    "x_doc_trust",
    "x_family_size",
)

FEATURE_NAMES: Tuple[str, ...] = SIGNAL_NAMES + EXTRA_FEATURE_NAMES


def _round(value: float, nd: int = 6) -> float:
    return float(round(float(value), nd))


@dataclass(frozen=True)
class TrustSnapshot:
    """Trust values **as they were at retrieval time**.

    Person A never mutates trust; it only reads a snapshot.  Snapshotting at
    retrieval time is what keeps training features free of future information
    (see plan Section 6.3, leakage guard).
    """

    doc_id: str
    source_id: str
    family_id: str
    t_doc: float = 0.5
    t_family: float = 0.5
    t_source: float = 0.5
    t_eff: float = 0.5
    status: TrustStatus = TrustStatus.TRUSTED
    n_doc_observations: int = 0
    as_of: float = field(default_factory=time.time)

    @classmethod
    def neutral(cls, doc_id: str, source_id: str = "unknown", family_id: str = "unknown") -> "TrustSnapshot":
        """Neutral snapshot used before Person B's ledger is wired in."""
        return cls(doc_id=doc_id, source_id=source_id, family_id=family_id)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class RetrievedDocument:
    """One retrieved passage, with provenance and trust attached."""

    doc_id: str
    chunk_id: str
    text: str
    similarity: float
    rank: int
    source_id: str
    family_id: str
    trust: TrustSnapshot
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "similarity": _round(self.similarity),
            "rank": int(self.rank),
            "source_id": self.source_id,
            "family_id": self.family_id,
            "trust": self.trust.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SignalVector:
    """The six cheap always-on signals, each in [0, 1] (higher = more suspicious)."""

    s1_query_echo: float = 0.0
    s2_similarity_outlier: float = 0.0
    s3_cluster_tightness: float = 0.0
    s4_ingestion_burst: float = 0.0
    s5_source_immaturity: float = 0.0
    s6_neighbourhood_density: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in SIGNAL_NAMES], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {n: _round(getattr(self, n)) for n in SIGNAL_NAMES}


@dataclass(frozen=True)
class FeatureSnapshot:
    """Full design row for the suspicion scorer, captured at retrieval time."""

    signals: SignalVector
    extras: Mapping[str, float]

    def as_array(self, feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
        values = []
        signal_dict = {n: getattr(self.signals, n) for n in SIGNAL_NAMES}
        for name in feature_names:
            if name in signal_dict:
                values.append(signal_dict[name])
            else:
                values.append(float(self.extras.get(name, 0.0)))
        return np.array(values, dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        out = self.signals.to_dict()
        out.update({k: _round(v) for k, v in self.extras.items()})
        return out


@dataclass(frozen=True)
class SecurityAssessment:
    """Person A's per-passage verdict handed to Person B's policy."""

    doc_id: str
    chunk_id: str
    query_id: str
    signals: SignalVector
    suspicion: float
    band: Band
    action: Action
    feature_snapshot: FeatureSnapshot
    scorer_version: str = "uncalibrated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "query_id": self.query_id,
            "signals": self.signals.to_dict(),
            "suspicion": _round(self.suspicion),
            "band": self.band.value,
            "action": self.action.value,
            "features": self.feature_snapshot.to_dict(),
            "scorer_version": self.scorer_version,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Produced by Person B's verifier; consumed here only for logging/abstention."""

    doc_id: str
    query_id: str
    single_doc_answer: Optional[str]
    influential: bool
    support_mass: float
    refute_mass: float
    outcome: VerificationOutcome
    llm_calls: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "query_id": self.query_id,
            "single_doc_answer": self.single_doc_answer,
            "influential": bool(self.influential),
            "support_mass": _round(self.support_mass),
            "refute_mass": _round(self.refute_mass),
            "outcome": self.outcome.value,
            "llm_calls": int(self.llm_calls),
            "latency_ms": _round(self.latency_ms, 3),
        }


@dataclass(frozen=True)
class PolicyDecision:
    """Person B decides what actually reaches the generator."""

    context_doc_ids: Tuple[str, ...]
    excluded_doc_ids: Tuple[str, ...] = ()
    verified: Tuple[VerificationResult, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_doc_ids": list(self.context_doc_ids),
            "excluded_doc_ids": list(self.excluded_doc_ids),
            "verified": [v.to_dict() for v in self.verified],
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    sentence: str
    supported: Optional[bool] = None   # filled by the NLI citation check when enabled
    nli_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "sentence": self.sentence,
            "supported": self.supported,
            "nli_score": None if self.nli_score is None else _round(self.nli_score),
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str
    llm_calls: int = 1
    latency_ms: float = 0.0
    model: str = "unknown"


@dataclass(frozen=True)
class AnswerRecord:
    """Everything needed for retroactive remediation and for Person C's metrics."""

    query_id: str
    query: str
    answer: str
    abstained: bool
    abstain_reason: Optional[str]
    citations: Tuple[Citation, ...]
    used_doc_ids: Tuple[str, ...]
    excluded_doc_ids: Tuple[str, ...]
    trust_snapshots: Mapping[str, Dict[str, Any]]
    evidence_mass: float
    llm_calls: int
    latency_ms: float
    step: int = 0                      # position in Person C's query stream
    timestamp: float = field(default_factory=time.time)
    model: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "abstained": bool(self.abstained),
            "abstain_reason": self.abstain_reason,
            "citations": [c.to_dict() for c in self.citations],
            "used_doc_ids": list(self.used_doc_ids),
            "excluded_doc_ids": list(self.excluded_doc_ids),
            "trust_snapshots": {k: dict(v) for k, v in self.trust_snapshots.items()},
            "evidence_mass": _round(self.evidence_mass),
            "llm_calls": int(self.llm_calls),
            "latency_ms": _round(self.latency_ms, 3),
            "step": int(self.step),
            "timestamp": _round(self.timestamp, 3),
            "model": self.model,
        }


# --------------------------------------------------------------------------
# Protocols: the integration surface
# --------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Dense encoder.  Returns L2-normalised float32 rows."""

    dim: int
    name: str

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


@runtime_checkable
class VectorIndex(Protocol):
    dim: int

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    def search(self, queries: np.ndarray, k: int, exclude: Optional[set] = None) -> List[List[Tuple[str, float]]]: ...
    def get_vector(self, id_: str) -> Optional[np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def __len__(self) -> int: ...


@runtime_checkable
class TrustProvider(Protocol):
    """Person B implements this against the SQLite trust ledger."""

    def get_trust(self, doc_ids: Sequence[str]) -> Dict[str, TrustSnapshot]: ...
    def blocked_doc_ids(self) -> set: ...


@runtime_checkable
class Verifier(Protocol):
    """Person B's corroboration-gated counterfactual verifier."""

    def verify(
        self,
        query: str,
        query_id: str,
        target: RetrievedDocument,
        pool: Sequence[RetrievedDocument],
    ) -> VerificationResult: ...


@runtime_checkable
class Policy(Protocol):
    """Person B's escalation policy / quarantine state machine."""

    def decide(
        self,
        query: str,
        query_id: str,
        documents: Sequence[RetrievedDocument],
        assessments: Sequence[SecurityAssessment],
        verifier: Optional[Verifier] = None,
    ) -> PolicyDecision: ...


@runtime_checkable
class LLM(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int = 256, stop: Optional[Sequence[str]] = None) -> LLMResponse: ...


# --------------------------------------------------------------------------
# Null implementations so Person A's pipeline runs before B exists
# --------------------------------------------------------------------------


class NullTrustProvider:
    """Neutral trust for every document; blocks nothing."""

    def __init__(self, default: float = 0.5) -> None:
        self.default = float(default)

    def get_trust(self, doc_ids: Sequence[str]) -> Dict[str, TrustSnapshot]:
        return {d: TrustSnapshot(doc_id=d, source_id="unknown", family_id="unknown",
                                 t_doc=self.default, t_family=self.default,
                                 t_source=self.default, t_eff=self.default)
                for d in doc_ids}

    def blocked_doc_ids(self) -> set:
        return set()


class NullVerifier:
    """Returns NEUTRAL without spending an LLM call."""

    def verify(self, query, query_id, target, pool) -> VerificationResult:  # type: ignore[no-untyped-def]
        return VerificationResult(
            doc_id=target.doc_id, query_id=query_id, single_doc_answer=None,
            influential=False, support_mass=0.0, refute_mass=0.0,
            outcome=VerificationOutcome.NEUTRAL, llm_calls=0, latency_ms=0.0,
        )


class DefaultPolicy:
    """Band-only fallback policy used until Person B's policy lands.

    LOW -> use, MEDIUM -> verify (if a verifier is supplied) then use unless
    refuted, HIGH -> exclude from this answer.  Mirrors plan Section 4.4.
    """

    def decide(self, query, query_id, documents, assessments, verifier=None):  # type: ignore[no-untyped-def]
        by_id = {a.doc_id: a for a in assessments}
        context: List[str] = []
        excluded: List[str] = []
        verified: List[VerificationResult] = []
        for doc in documents:
            assessment = by_id.get(doc.doc_id)
            band = assessment.band if assessment else Band.LOW
            if band is Band.HIGH:
                excluded.append(doc.doc_id)
                continue
            if band is Band.MEDIUM and verifier is not None:
                result = verifier.verify(query, query_id, doc, documents)
                verified.append(result)
                if result.outcome is VerificationOutcome.REFUTE:
                    excluded.append(doc.doc_id)
                    continue
            context.append(doc.doc_id)
        return PolicyDecision(
            context_doc_ids=tuple(context),
            excluded_doc_ids=tuple(excluded),
            verified=tuple(verified),
            notes={"policy": "DefaultPolicy"},
        )
