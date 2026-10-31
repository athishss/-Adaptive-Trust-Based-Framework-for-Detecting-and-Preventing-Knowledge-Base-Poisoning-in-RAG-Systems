"""Escalation policy with trust-ledger integration (plan Section 4.4).

This is Person B's security state machine.  Risk band, verification outcome and
persistent trust state are deliberately separate concepts:

===================  ==========================  ==============================
band                 action for this answer      trust-state consequence
===================  ==========================  ==============================
LOW                  use                         none
MEDIUM              verify (inline):            one observation recorded,
                     SUPPORT -> use              deduplicated per query
                     REFUTE  -> exclude
                     NEUTRAL -> use, but only
                       when a verifier ran
                     no verifier -> exclude
                       (fail closed)
HIGH                 exclude and queue           passive beta penalty, then
                                                 off-path verification
QUARANTINED/REJECTED  never retrieved            (TrustProvider contract)
===================  ==========================  ==============================

A document can therefore be *unresolved* (MEDIUM, queued, verification failed)
without being called trusted.  The policy never turns "we do not know" into
"this is clean".

Inline verification and queued verification share :meth:`_apply_outcome`, so a
queued REFUTE triggers the same ledger update, quarantine transition and
retroactive remediation callback as an inline one.  That divergence was a real
defect: queued refutations used to change trust but never flag the answers that
had already used the passage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..contracts import (
    Band,
    PolicyDecision,
    RetrievedDocument,
    SecurityAssessment,
    VerificationOutcome,
    VerificationResult,
    Verifier,
)
from ..utils.logging import get_logger
from .ledger import TrustLedger
from .queue import VerificationQueue
from .verifier import CorroborationVerifier

logger = get_logger(__name__)


@dataclass
class QueueDrainReport:
    """What one off-path drain did, including remediation triggers."""

    results: List[VerificationResult] = field(default_factory=list)
    newly_quarantined: List[str] = field(default_factory=list)
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "newly_quarantined": list(self.newly_quarantined),
            "errors": int(self.errors),
        }


class TrustPolicy:
    """Band-based escalation policy with trust updates and quarantine.

    Satisfies the ``Policy`` protocol from ``contracts.py``.

    Parameters
    ----------
    ledger : TrustLedger
        The trust ledger to update after each verification.
    on_quarantine : callable, optional
        Callback ``f(doc_ids, reason)`` invoked when passages are quarantined.
        Typically set to ``pipeline.on_quarantine`` (retroactive remediation).
    queue : VerificationQueue, optional
        When supplied, every HIGH-band passage is queued for verification off
        the latency path.
    fail_open_without_verifier : bool
        Research-only escape hatch.  When False (default) a MEDIUM-band passage
        is excluded if no verifier is available, because the policy says MEDIUM
        requires corroboration.
    """

    def __init__(self, ledger: TrustLedger,
                 on_quarantine: Optional[Callable] = None,
                 queue: Optional[VerificationQueue] = None,
                 fail_open_without_verifier: bool = False,
                 use_on_neutral: bool = False) -> None:
        self.ledger = ledger
        self._on_quarantine = on_quarantine
        self.queue = queue
        self.fail_open_without_verifier = bool(fail_open_without_verifier)
        self.use_on_neutral = bool(use_on_neutral)
        self.last_drain: Optional[QueueDrainReport] = None
        self.last_decisions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    #  Policy protocol
    # ------------------------------------------------------------------

    def decide(self, query: str, query_id: str,
               documents: Sequence[RetrievedDocument],
               assessments: Sequence[SecurityAssessment],
               verifier: Optional[Verifier] = None,
               now: Optional[float] = None,
               **kwargs) -> PolicyDecision:
        """Decide which passages reach the generator for this answer."""
        by_id: Dict[str, SecurityAssessment] = {a.doc_id: a for a in assessments}
        pool = kwargs.get("pool", documents)
        ts = self.ledger.now() if now is None else float(now)
        # Trust decay is scheduled per query; decide() is called once per query.
        self.ledger.begin_query(ts)

        context: List[str] = []
        excluded: List[str] = []
        verified: List[VerificationResult] = []
        newly_quarantined: List[str] = []
        decisions: Dict[str, Dict[str, Any]] = {}

        for doc in documents:
            assessment = by_id.get(doc.doc_id)
            band = assessment.band if assessment else Band.LOW
            status_before = self.ledger.get_status(doc.doc_id)
            record: Dict[str, Any] = {"band": band.value,
                                      "status_before": status_before.value,
                                      "outcome": None, "reason": None}

            if doc.trust.blocked or status_before.value in ("QUARANTINED", "REJECTED"):
                # Defence in depth: a blocked passage should never be in the
                # retrieved set, but policy must not pass it through if it is.
                excluded.append(doc.doc_id)
                record["reason"] = "blocked_status"
                decisions[doc.doc_id] = record
                continue

            if band is Band.HIGH:
                excluded.append(doc.doc_id)
                record["reason"] = "high_band_excluded"
                self.ledger.record_high_band(
                    doc.doc_id, doc.source_id, doc.family_id, timestamp=ts,
                )
                if self.queue is not None:
                    self.queue.enqueue(query, query_id, doc, pool,
                                       reason="HIGH band", timestamp=ts)
                decisions[doc.doc_id] = record
                continue

            if band is Band.MEDIUM:
                if verifier is None:
                    if self.fail_open_without_verifier:
                        context.append(doc.doc_id)
                        record["reason"] = "medium_no_verifier_fail_open"
                    else:
                        excluded.append(doc.doc_id)
                        record["reason"] = "medium_no_verifier_fail_closed"
                    decisions[doc.doc_id] = record
                    continue

                result = self._verify_one(query, query_id, doc, pool, verifier, ts)
                verified.append(result)
                record["outcome"] = result.outcome.value
                if result.error:
                    record["reason"] = f"verification_error: {result.error}"
                applied, quarantined = self._apply_outcome(
                    doc, result, query_id, ts,
                )
                if quarantined:
                    newly_quarantined.append(doc.doc_id)
                if result.outcome is VerificationOutcome.REFUTE or quarantined:
                    excluded.append(doc.doc_id)
                    record["reason"] = record["reason"] or "refuted"
                elif result.outcome is VerificationOutcome.NEUTRAL and not self.use_on_neutral:
                    # Unresolved is not support.  The passage is withheld from
                    # this answer without a trust penalty, so a false positive
                    # is reversible and does not quarantine clean evidence.
                    excluded.append(doc.doc_id)
                    record["reason"] = record["reason"] or "neutral_unresolved"
                else:
                    context.append(doc.doc_id)
                    record["reason"] = record["reason"] or "verified"
                record["trust_applied"] = applied
                decisions[doc.doc_id] = record
                continue

            # LOW band: use as ordinary evidence.
            context.append(doc.doc_id)
            record["reason"] = "low_band_used"
            decisions[doc.doc_id] = record

        self._remediate(newly_quarantined, "refuted by corroboration-gated verifier", ts)
        self.last_decisions = decisions

        notes: Dict[str, Any] = {
            "policy": "TrustPolicy",
            "newly_quarantined": newly_quarantined,
            "n_verified": len(verified),
            "n_excluded": len(excluded),
            "decisions": decisions,
            "timestamp": ts,
        }
        return PolicyDecision(
            context_doc_ids=tuple(context),
            excluded_doc_ids=tuple(excluded),
            verified=tuple(verified),
            notes=notes,
        )

    def drain_verification_queue(self, verifier: Optional[Verifier] = None,
                                 now: Optional[float] = None,
                                 max_items: Optional[int] = None) -> List[VerificationResult]:
        """Verify queued HIGH-band passages off the latency path.

        Every outcome goes through the same ledger update and quarantine path as
        inline verification, and newly quarantined documents trigger the same
        remediation callback.  The detailed report is kept in
        ``self.last_drain``.
        """
        if self.queue is None:
            self.last_drain = QueueDrainReport()
            return []
        ts = self.ledger.now() if now is None else float(now)
        before = set(self.ledger.blocked_doc_ids())
        results: List[VerificationResult] = []
        errors = 0
        # Snapshot first: draining pops items, and the callback needs the docs.
        pending = list(self.queue.pending())[: max_items] if max_items else list(self.queue.pending())
        if not pending:
            self.last_drain = QueueDrainReport()
            return []

        if isinstance(verifier, CorroborationVerifier):
            results = self.queue.drain(
                verifier, self.ledger, now=ts, max_items=len(pending),
                same_burst=self.ledger.same_burst,
                source_influence=lambda src: self.ledger.influence_factor(src, ts),
            )
        elif verifier is not None:
            results = self.queue.drain(verifier, self.ledger, now=ts, max_items=len(pending))
        errors = int(getattr(self.queue, "errors", 0))
        after = set(self.ledger.blocked_doc_ids())
        newly = sorted(after - before)
        self._remediate(newly, "quarantined after off-path verification", ts)
        self.last_drain = QueueDrainReport(results=list(results), newly_quarantined=newly,
                                           errors=errors)
        return results

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _verify_one(self, query: str, query_id: str, doc: RetrievedDocument,
                    pool: Sequence[RetrievedDocument], verifier: Verifier,
                    ts: float) -> VerificationResult:
        """Run one verification; a backend failure becomes an explicit NEUTRAL.

        A crashed verifier must not crash the query, and it must not silently
        grant trust either: the result below is NEUTRAL with ``error`` set, and
        the passage stays unresolved (it is used only because the answer needs
        context; it earns no trust).
        """
        try:
            if isinstance(verifier, CorroborationVerifier):
                result = verifier.verify(
                    query, query_id, doc, pool,
                    same_burst=self.ledger.same_burst,
                    source_influence=lambda src: self.ledger.influence_factor(src, ts),
                )
            else:
                result = verifier.verify(query, query_id, doc, pool)
        except Exception as exc:  # noqa: BLE001 - a backend failure is data, not a crash
            logger.warning("verification failed for %s: %s", doc.doc_id, exc, exc_info=True)
            return VerificationResult(
                doc_id=doc.doc_id, query_id=query_id, single_doc_answer=None,
                influential=False, support_mass=0.0, refute_mass=0.0,
                outcome=VerificationOutcome.NEUTRAL, llm_calls=0, latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(result, VerificationResult):
            logger.warning("verifier returned %r for %s; treating as NEUTRAL",
                           type(result).__name__, doc.doc_id)
            return VerificationResult(
                doc_id=doc.doc_id, query_id=query_id, single_doc_answer=None,
                influential=False, support_mass=0.0, refute_mass=0.0,
                outcome=VerificationOutcome.NEUTRAL, llm_calls=0, latency_ms=0.0,
                error=f"malformed verifier result: {type(result).__name__}",
            )
        if result.doc_id != doc.doc_id:
            logger.warning("verifier result doc_id %s does not match target %s",
                           result.doc_id, doc.doc_id)
            return VerificationResult(
                doc_id=doc.doc_id, query_id=query_id,
                single_doc_answer=result.single_doc_answer,
                influential=bool(result.influential),
                support_mass=float(result.support_mass),
                refute_mass=float(result.refute_mass),
                outcome=VerificationOutcome.NEUTRAL,
                llm_calls=int(result.llm_calls), latency_ms=float(result.latency_ms),
                mode=result.mode,
                error="verifier returned a result for a different document",
            )
        if not isinstance(result.outcome, VerificationOutcome):
            logger.warning("verifier returned outcome %r for %s; treating as NEUTRAL",
                           result.outcome, doc.doc_id)
            return VerificationResult(
                doc_id=doc.doc_id, query_id=query_id,
                single_doc_answer=result.single_doc_answer,
                influential=bool(result.influential),
                support_mass=float(result.support_mass),
                refute_mass=float(result.refute_mass),
                outcome=VerificationOutcome.NEUTRAL,
                llm_calls=int(result.llm_calls), latency_ms=float(result.latency_ms),
                mode=result.mode, evidence_doc_ids=tuple(result.evidence_doc_ids),
                error="verifier returned a malformed outcome",
            )
        return result

    def _apply_outcome(self, doc: RetrievedDocument, result: VerificationResult,
                       query_id: str, ts: float) -> tuple:
        """Record one outcome exactly once; return (applied, now_quarantined)."""
        if result.error:
            # Do not move trust on a failed verification.
            return False, False
        evidence_key = f"{query_id}|{doc.doc_id}|{result.outcome.value}"
        before = self.ledger.get_status(doc.doc_id)
        applied = self.ledger.record_observation(
            doc.doc_id, doc.source_id, doc.family_id,
            outcome=result.outcome.value, timestamp=ts, evidence_key=evidence_key,
        )
        after = self.ledger.get_status(doc.doc_id)
        quarantined = before is not after and after.value in ("QUARANTINED", "REJECTED")
        return bool(applied), bool(quarantined)

    def _remediate(self, doc_ids: Sequence[str], reason: str, ts: float) -> None:
        """Invoke the remediation callback for newly quarantined documents.

        Inline and queued verification both call this function, so the two
        paths cannot drift apart.  A failing callback is logged, never allowed
        to break the query.
        """
        if not doc_ids or self._on_quarantine is None:
            return
        try:
            if _accepts_at(self._on_quarantine):
                self._on_quarantine(list(doc_ids), reason=reason, at=ts)
            else:
                self._on_quarantine(list(doc_ids), reason=reason)
        except TypeError:
            # Callbacks written against the older two-argument contract.
            try:
                self._on_quarantine(list(doc_ids), reason)
            except Exception:  # noqa: BLE001
                logger.warning("on_quarantine callback failed for %s", doc_ids,
                               exc_info=True)
        except Exception:  # noqa: BLE001
            logger.warning("on_quarantine callback failed for %s", doc_ids,
                           exc_info=True)


def _accepts_at(callback: Callable) -> bool:
    """Whether a quarantine callback accepts the optional ``at`` keyword."""
    import inspect

    try:
        return "at" in inspect.signature(callback).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
