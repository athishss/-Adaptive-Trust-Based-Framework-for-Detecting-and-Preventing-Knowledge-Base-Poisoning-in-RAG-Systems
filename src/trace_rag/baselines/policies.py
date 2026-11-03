"""Baselines for Person C evaluation, runnable through the pipeline interface.

Every advertised baseline exists here and consumes the same canonical contracts
as the protected system:

* documents -> ``RetrievedDocument`` (trust snapshot is ``t_eff``; there is no
  ``trust.trust`` field - that mismatch used to make two baselines crash);
* decisions -> ``PolicyDecision``;
* verification -> the same ``Verifier`` protocol.

``perplexity`` is an order-2 language model fitted on the *clean* corpus and
scored per token.  It is a real, reproducible baseline (not a placeholder), but
it is a lexical surrogate for the perplexity filters in the literature, and the
experiment report says so.

No baseline reads ground-truth labels.  Labels enter only at metric time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from trace_rag.contracts import (
    Band,
    PolicyDecision,
    RetrievedDocument,
    SecurityAssessment,
    VerificationOutcome,
    VerificationResult,
)
from trace_rag.trust.verifier import CorroborationVerifier
from trace_rag.utils.textnorm import normalise, tokenise


@dataclass(frozen=True)
class BaselineResult:
    documents: List[RetrievedDocument]
    blocked: List[str]
    verified: List[VerificationResult]
    llm_calls: int = 0
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "blocked": list(self.blocked),
            "verified": [v.to_dict() for v in self.verified],
            "llm_calls": int(self.llm_calls),
            "notes": dict(self.notes),
        }


def no_defence(documents: Sequence[RetrievedDocument]) -> BaselineResult:
    return BaselineResult(documents=list(documents), blocked=[], verified=[], llm_calls=0,
                          notes={"policy": "no_defence"})


def duplicate_filter(documents: Sequence[RetrievedDocument]) -> BaselineResult:
    """Drop passages whose normalised text has already been seen."""
    seen = set()
    kept: List[RetrievedDocument] = []
    blocked: List[str] = []
    for doc in documents:
        key = normalise(doc.text).lower()
        if key in seen:
            blocked.append(doc.doc_id)
        else:
            seen.add(key)
            kept.append(doc)
    return BaselineResult(documents=kept, blocked=blocked, verified=[], llm_calls=0,
                          notes={"policy": "duplicate_filter"})


class NGramPerplexityModel:
    """Deterministic order-``n`` language model over clean passages.

    Scores are mean negative log-likelihood per token with add-``smoothing``
    smoothing and backoff to the unigram distribution.  Higher means more
    surprising, which is what a perplexity filter rejects.
    """

    def __init__(self, order: int = 2, smoothing: float = 1.0) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.order = int(order)
        self.smoothing = float(smoothing)
        self._counts: Dict[tuple, int] = {}
        self._vocabulary: set = set()
        self._n_tokens = 0

    def fit(self, texts: Sequence[str]) -> "NGramPerplexityModel":
        for text in texts:
            tokens = ["<s>"] + tokenise(text) + ["</s>"]
            self._vocabulary.update(tokens)
            for i, token in enumerate(tokens):
                self._n_tokens += 1
                for n in range(1, self.order + 1):
                    if i - n + 1 < 0:
                        continue
                    self._counts[tuple(tokens[i - n + 1:i + 1])] = \
                        self._counts.get(tuple(tokens[i - n + 1:i + 1]), 0) + 1
        if self._n_tokens == 0:
            raise ValueError("cannot fit a perplexity model on an empty corpus")
        return self

    def score(self, text: str) -> float:
        """Mean NLL per token; ``inf`` for an empty passage."""
        tokens = tokenise(text)
        if not tokens:
            return float("inf")
        sequence = ["<s>"] + tokens + ["</s>"]
        vocab = max(1, len(self._vocabulary))
        total = 0.0
        for i in range(1, len(sequence)):
            probability = None
            for n in range(min(self.order, i), 0, -1):
                context = tuple(sequence[i - n:i])
                count = self._counts.get(context + (sequence[i],), 0)
                context_count = self._counts.get(context, 0)
                if context_count:
                    probability = (count + self.smoothing) / (
                        context_count + self.smoothing * vocab)
                    break
            if probability is None:
                probability = 1.0 / vocab
            total -= math.log(max(probability, 1e-12))
        return total / len(tokens)


def perplexity_filter(documents: Sequence[RetrievedDocument],
                      model: NGramPerplexityModel, threshold: float) -> BaselineResult:
    """Flag passages whose LM score exceeds ``threshold``."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    kept, blocked = [], []
    for doc in documents:
        if model.score(doc.text) > threshold:
            blocked.append(doc)
        else:
            kept.append(doc)
    return BaselineResult(documents=kept, blocked=[d.doc_id for d in blocked],
                          verified=[], llm_calls=0,
                          notes={"policy": "perplexity", "threshold": threshold})


def trust_threshold_filter(documents: Sequence[RetrievedDocument],
                           threshold: float) -> BaselineResult:
    """Keep passages whose effective trust is at least ``threshold``.

    The canonical field is ``t_eff``.  Reading ``doc.trust.trust`` (which never
    existed on ``TrustSnapshot``) made this baseline crash instead of running.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    kept, blocked = [], []
    for doc in documents:
        if float(doc.trust.t_eff) < float(threshold):
            blocked.append(doc.doc_id)
        else:
            kept.append(doc)
    return BaselineResult(documents=kept, blocked=blocked, verified=[], llm_calls=0,
                          notes={"policy": "trust_threshold", "threshold": float(threshold)})


def always_on_loo(query: str, query_id: str,
                  documents: Sequence[RetrievedDocument],
                  verifier: CorroborationVerifier) -> BaselineResult:
    """Verify every passage, not just the suspicious ones.

    This is the cost reference for corroboration-gated verification, so it must
    report its real LLM-call count and use the shared enum, not a string.
    """
    docs = list(documents)
    kept, blocked, verified = [], [], []
    llm_calls = 0
    for target in docs:
        result = verifier.verify(query=query, query_id=query_id, target=target, pool=docs)
        verified.append(result)
        llm_calls += int(result.llm_calls)
        if result.outcome is VerificationOutcome.REFUTE:
            blocked.append(target.doc_id)
        else:
            kept.append(target)
    return BaselineResult(documents=kept, blocked=blocked, verified=verified,
                          llm_calls=llm_calls, notes={"policy": "always_on_loo"})


# ---------------------------------------------------------------------------
#  Policies: the same interface the protected system implements
# ---------------------------------------------------------------------------


class _BaseBaselinePolicy:
    name = "baseline"

    def decide(self, query: str, query_id: str, documents: Sequence[RetrievedDocument],
               assessments: Sequence[SecurityAssessment], verifier: Any = None,
               now: Optional[float] = None, **kwargs: Any) -> PolicyDecision:
        result = self.apply(query, query_id, documents, assessments, verifier, **kwargs)
        return PolicyDecision(
            context_doc_ids=tuple(d.doc_id for d in result.documents),
            excluded_doc_ids=tuple(result.blocked),
            verified=tuple(result.verified),
            notes={**dict(result.notes), "policy": self.name,
                   "n_llm_calls": int(result.llm_calls),
                   "decisions": {d.doc_id: {"reason": self.name} for d in documents}},
        )

    def apply(self, query: str, query_id: str, documents: Sequence[RetrievedDocument],
              assessments: Sequence[SecurityAssessment], verifier: Any = None,
              **kwargs: Any) -> BaselineResult:
        raise NotImplementedError


class NoDefencePolicy(_BaseBaselinePolicy):
    name = "no_defence"

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        return no_defence(documents)


class DuplicateFilterPolicy(_BaseBaselinePolicy):
    name = "duplicate_filter"

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        return duplicate_filter(documents)


class TrustThresholdPolicy(_BaseBaselinePolicy):
    name = "trust_threshold"

    def __init__(self, threshold: float = 0.20) -> None:
        self.threshold = float(threshold)

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        return trust_threshold_filter(documents, self.threshold)


class PerplexityPolicy(_BaseBaselinePolicy):
    name = "perplexity"

    def __init__(self, model: NGramPerplexityModel, threshold: float) -> None:
        self.model = model
        self.threshold = float(threshold)

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        return perplexity_filter(documents, self.model, self.threshold)


class AlwaysOnLOOPolicy(_BaseBaselinePolicy):
    name = "always_on_loo"

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        if verifier is None:
            raise ValueError("always_on_loo needs a verifier")
        return always_on_loo(query, query_id, documents, verifier)


class TrustRAGPolicy:
    """Simplified TrustRAG baseline: trust floor plus near-duplicate dedup.

    The published TrustRAG line of work clusters retrieved passages and trusts
    clusters by their supporting sources.  This is a deliberately small,
    label-free adaptation of that idea for the shared harness: keep passages
    whose effective trust clears ``trust_floor`` and keep at most one passage
    per near-duplicate family.  It **does not verify** suspicious passages and
    never updates the ledger, so it is a genuine no-verification baseline, not
    another spelling of the full RAGLedger policy.
    """

    name = "TrustRAG"

    def __init__(self, ledger: Any = None, trust_floor: float = 0.20,
                 on_quarantine: Any = None, queue: Any = None) -> None:
        if not 0.0 <= float(trust_floor) <= 1.0:
            raise ValueError("trust_floor must be in [0, 1]")
        self.ledger = ledger
        self.trust_floor = float(trust_floor)
        # Accepted for interface compatibility; this baseline has no queue.
        self.on_quarantine = on_quarantine
        self.queue = queue

    def decide(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        kept: List[RetrievedDocument] = []
        excluded: List[str] = []
        seen_families = set()
        decisions: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            trust = float(doc.trust.t_eff) if doc.trust is not None else 0.0
            if trust < self.trust_floor:
                excluded.append(doc.doc_id)
                decisions[doc.doc_id] = {"reason": "low_trust"}
            elif doc.family_id in seen_families:
                excluded.append(doc.doc_id)
                decisions[doc.doc_id] = {"reason": "duplicate_family"}
            else:
                seen_families.add(doc.family_id)
                kept.append(doc)
                decisions[doc.doc_id] = {"reason": "trusted_family_representative"}
        return PolicyDecision(
            context_doc_ids=tuple(doc.doc_id for doc in kept),
            excluded_doc_ids=tuple(excluded),
            verified=(),
            notes={"policy": "TrustRAG", "trust_floor": self.trust_floor,
                   "n_llm_calls": 0, "decisions": decisions},
        )


class RobustRAGPolicy:
    """Simplified RobustRAG baseline: band blocking without verification.

    Keeps the trust floor and drops HIGH-band passages, but never corroborates
    a MEDIUM-band passage and never queues one for later verification.  This
    isolates the value of verification relative to a purely static filter.
    """

    name = "RobustRAG"

    def __init__(self, ledger: Any = None, trust_floor: float = 0.20,
                 on_quarantine: Any = None, queue: Any = None) -> None:
        if not 0.0 <= float(trust_floor) <= 1.0:
            raise ValueError("trust_floor must be in [0, 1]")
        self.ledger = ledger
        self.trust_floor = float(trust_floor)
        self.on_quarantine = on_quarantine
        self.queue = queue

    def decide(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        bands = {a.doc_id: a.band for a in assessments}
        kept: List[RetrievedDocument] = []
        excluded: List[str] = []
        decisions: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            trust = float(doc.trust.t_eff) if doc.trust is not None else 0.0
            band = bands.get(doc.doc_id, Band.LOW)
            if trust < self.trust_floor:
                excluded.append(doc.doc_id)
                decisions[doc.doc_id] = {"reason": "low_trust"}
            elif band is Band.HIGH:
                excluded.append(doc.doc_id)
                decisions[doc.doc_id] = {"reason": "high_band_no_verification"}
            else:
                kept.append(doc)
                decisions[doc.doc_id] = {"reason": f"{band.value.lower()}_band_used"}
        return PolicyDecision(
            context_doc_ids=tuple(doc.doc_id for doc in kept),
            excluded_doc_ids=tuple(excluded),
            verified=(),
            notes={"policy": "RobustRAG", "trust_floor": self.trust_floor,
                   "n_llm_calls": 0, "decisions": decisions},
        )


class SuspicionOnlyPolicy(_BaseBaselinePolicy):
    """Use the six-signal bands, but no trust state and no verification."""

    name = "suspicion_only"

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        bands = {a.doc_id: a.band for a in assessments}
        kept, excluded = [], []
        for doc in documents:
            if bands.get(doc.doc_id) is Band.HIGH:
                excluded.append(doc.doc_id)
            else:
                kept.append(doc)
        return BaselineResult(
            documents=kept, blocked=excluded, verified=[], llm_calls=0,
            notes={"policy": self.name, "uses_trust": False,
                   "uses_verification": False},
        )


class CorroborationOnlyPolicy(_BaseBaselinePolicy):
    """Verify every suspicious document; no hierarchical trust or quarantine."""

    name = "corroboration_only"

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        if verifier is None:
            raise ValueError("corroboration_only needs a verifier")
        bands = {a.doc_id: a.band for a in assessments}
        kept, excluded, verified = [], [], []
        calls = 0
        for doc in documents:
            if bands.get(doc.doc_id, Band.LOW) is Band.LOW:
                kept.append(doc)
                continue
            result = verifier.verify(query, query_id, doc, list(documents))
            verified.append(result)
            calls += int(result.llm_calls)
            if result.outcome is VerificationOutcome.REFUTE:
                excluded.append(doc.doc_id)
            elif result.outcome is VerificationOutcome.NEUTRAL:
                excluded.append(doc.doc_id)       # fail closed, no trust update
            else:
                kept.append(doc)
        return BaselineResult(
            documents=kept, blocked=excluded, verified=verified, llm_calls=calls,
            notes={"policy": self.name, "uses_trust": False,
                   "uses_verification": True, "deferred": False},
        )


class TrustNoDeferredPolicy(_BaseBaselinePolicy):
    """Trust floor plus inline verification; no queue or historical remediation."""

    name = "trust_no_deferred"

    def __init__(self, threshold: float = 0.20) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = float(threshold)

    def apply(self, query, query_id, documents, assessments, verifier=None, **kwargs):
        if verifier is None:
            raise ValueError("trust_no_deferred needs a verifier")
        bands = {a.doc_id: a.band for a in assessments}
        kept, excluded, verified = [], [], []
        calls = 0
        for doc in documents:
            if float(doc.trust.t_eff) < self.threshold:
                excluded.append(doc.doc_id)
                continue
            if bands.get(doc.doc_id) is Band.HIGH:
                excluded.append(doc.doc_id)
                continue
            if bands.get(doc.doc_id) is Band.MEDIUM:
                result = verifier.verify(query, query_id, doc, list(documents))
                verified.append(result)
                calls += int(result.llm_calls)
                if result.outcome is not VerificationOutcome.SUPPORT:
                    excluded.append(doc.doc_id)
                    continue
            kept.append(doc)
        return BaselineResult(
            documents=kept, blocked=excluded, verified=verified, llm_calls=calls,
            notes={"policy": self.name, "uses_trust": True,
                   "uses_verification": True, "deferred": False},
        )


BASELINE_SPECS = {
    "no_defence": {"published_method": False, "uses_trust": False,
                   "uses_verification": False, "deferred": False},
    "duplicate_filter": {"published_method": False, "uses_trust": False,
                         "uses_verification": False, "deferred": False},
    "perplexity": {"published_method": False, "uses_trust": False,
                   "uses_verification": False, "deferred": False,
                   "simplification": "order-2 n-gram surrogate, not neural perplexity"},
    "suspicion_only": {"published_method": False, "uses_trust": False,
                       "uses_verification": False, "deferred": False},
    "corroboration_only": {"published_method": False, "uses_trust": False,
                            "uses_verification": True, "deferred": False},
    "trust_threshold": {"published_method": False, "uses_trust": True,
                        "uses_verification": False, "deferred": False},
    "trust_no_deferred": {"published_method": False, "uses_trust": True,
                          "uses_verification": True, "deferred": False},
    "TrustRAG": {"published_method": False, "uses_trust": True,
                 "uses_verification": False, "deferred": False,
                 "simplification": "label-free adaptation, not the published algorithm"},
    "RobustRAG": {"published_method": False, "uses_trust": True,
                  "uses_verification": False, "deferred": False,
                  "simplification": "band-blocking adaptation, not certified aggregation"},
    "always_on_loo": {"published_method": False, "uses_trust": False,
                      "uses_verification": True, "deferred": False},
    "RAGLedger": {"published_method": False, "uses_trust": True,
                  "uses_verification": True, "deferred": True},
}

BASELINE_NAMES = tuple(name for name in BASELINE_SPECS if name != "RAGLedger")

BASELINE_POLICIES = {
    "no_defence": NoDefencePolicy,
    "perplexity": PerplexityPolicy,
    "duplicate_filter": DuplicateFilterPolicy,
    "trust_threshold": TrustThresholdPolicy,
    "TrustRAG": TrustRAGPolicy,
    "RobustRAG": RobustRAGPolicy,
    "suspicion_only": SuspicionOnlyPolicy,
    "corroboration_only": CorroborationOnlyPolicy,
    "trust_no_deferred": TrustNoDeferredPolicy,
    "always_on_loo": AlwaysOnLOOPolicy,
}


def build_baseline_policy(name: str, *, ledger: Any = None,
                          on_quarantine: Any = None, queue: Any = None,
                          trust_floor: float = 0.20,
                          perplexity_model: Optional[NGramPerplexityModel] = None,
                          perplexity_threshold: float = 8.0) -> Any:
    """Instantiate an advertised baseline by name.

    Raises a clear error when a baseline is asked for without the state it
    requires (for example TrustRAG without a ledger) instead of silently
    skipping the case.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; available: {BASELINE_NAMES}")
    if name in ("TrustRAG", "RobustRAG"):
        # These baselines read the trust snapshot attached to each retrieved
        # document; a live ledger is optional and is never updated by them.
        if name == "TrustRAG":
            return TrustRAGPolicy(ledger=ledger, trust_floor=trust_floor,
                                  on_quarantine=on_quarantine, queue=queue)
        return RobustRAGPolicy(ledger=ledger, trust_floor=trust_floor,
                               on_quarantine=on_quarantine, queue=queue)
    if name == "perplexity":
        if perplexity_model is None:
            raise ValueError("perplexity baseline requires a fitted NGramPerplexityModel")
        return PerplexityPolicy(perplexity_model, perplexity_threshold)
    if name == "trust_threshold":
        return TrustThresholdPolicy(threshold=trust_floor)
    if name == "trust_no_deferred":
        return TrustNoDeferredPolicy(threshold=trust_floor)
    return BASELINE_POLICIES[name]()
