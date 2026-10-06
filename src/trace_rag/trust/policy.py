"""Escalation policy with trust-ledger integration (plan Section 4.4).

Extends the band-based decision logic of ``DefaultPolicy`` with:
  * trust-ledger updates after every verification;
  * automatic quarantine when the state machine transitions;
  * a passive HIGH-band beta penalty for passages too suspicious to verify;
  * the async verification queue: HIGH-band passages are excluded right away
    and queued, then verified off the latency path when the caller drains
    (or a background worker runs);
  * reporting of newly quarantined doc_ids in the decision notes.

The caller (typically Person C's harness or the pipeline user) reads
``decision.notes["newly_quarantined"]`` and calls
``pipeline.on_quarantine(...)`` to trigger retroactive remediation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..contracts import (
    Band,
    PolicyDecision,
    RetrievedDocument,
    SecurityAssessment,
    TrustStatus,
    VerificationOutcome,
    VerificationResult,
    Verifier,
)
from ..utils.logging import get_logger
from .ledger import TrustLedger
from .queue import VerificationQueue
from .verifier import CorroborationVerifier

logger = get_logger(__name__)


class TrustPolicy:
    """Band-based escalation policy with trust updates and quarantine.

    Satisfies the ``Policy`` protocol from ``contracts.py``.

    Decision logic (per passage):

      * **LOW** → use as context, no further action.
      * **MEDIUM** → verify (if a verifier is supplied), then:
          - SUPPORT → use, bump trust positively;
          - REFUTE  → exclude, bump trust negatively, check quarantine;
          - NEUTRAL → use (benefit of the doubt).
      * **HIGH** → exclude, apply a mild passive trust penalty.

    Parameters
    ----------
    ledger : TrustLedger
        The trust ledger to update after each verification.
    on_quarantine : callable, optional
        Callback ``f(doc_ids, reason)`` invoked when passages are quarantined.
        Typically set to ``pipeline.on_quarantine``.
    queue : VerificationQueue, optional
        When supplied, every HIGH-band passage is queued for verification off
        the latency path (plan Section 4.4).  The caller drains the queue with
        ``drain_verification_queue`` or runs it with ``start_background``.
    """

    def __init__(self, ledger: TrustLedger,
                 on_quarantine: Optional[Callable] = None,
                 queue: Optional[VerificationQueue] = None) -> None:
        self.ledger = ledger
        self._on_quarantine = on_quarantine
        self.queue = queue if queue is not None else VerificationQueue(on_quarantine=on_quarantine)
        if on_quarantine is not None and self.queue.on_quarantine is None:
            self.queue.on_quarantine = on_quarantine

    # ------------------------------------------------------------------
    #  Policy protocol
    # ------------------------------------------------------------------

    def decide(self, query: str, query_id: str,
               documents: Sequence[RetrievedDocument],
               assessments: Sequence[SecurityAssessment],
               verifier: Optional[Verifier] = None,
               **kwargs) -> PolicyDecision:
        """Decide which passages reach the generator.

        Returns a ``PolicyDecision`` with:
          * ``context_doc_ids`` — passages allowed into the answer;
          * ``excluded_doc_ids`` — passages withheld;
          * ``verified`` — all ``VerificationResult`` objects produced;
          * ``notes`` — includes ``newly_quarantined`` list.
        """
        by_id: Dict[str, SecurityAssessment] = {a.doc_id: a for a in assessments}
        pool = kwargs.get("pool", documents)  # full candidate pool for corroboration
        now_value = kwargs.get("now")
        now = time.time() if now_value is None else float(now_value)
        # Trust decay is scheduled per query, and decide() is called once by the pipeline.
        self.ledger.begin_query(now)

        # Retrieval normally materialises these rows through get_trust(). Keep
        # the policy correct for direct/API callers too: register the supplied
        # pool as provenance-only NEUTRAL observations before source influence
        # or burst checks. NEUTRAL does not change any Beta counts or trust.
        for doc in pool:
            ingested_at = doc.metadata.get("ingested_at", now)
            try:
                registration_time = float(ingested_at)
            except (TypeError, ValueError):
                registration_time = now
            self.ledger.record_observation(
                doc.doc_id, doc.source_id, doc.family_id,
                outcome=VerificationOutcome.NEUTRAL.value,
                timestamp=registration_time,
            )

        context: List[str] = []
        excluded: List[str] = []
        verified: List[VerificationResult] = []
        newly_quarantined: List[str] = []
        unavailable: List[str] = []

        for doc in documents:
            status_before = doc.trust.status
            if status_before in (TrustStatus.QUARANTINED, TrustStatus.REJECTED):
                excluded.append(doc.doc_id)
                continue

            assessment = by_id.get(doc.doc_id)
            band = assessment.band if assessment else Band.LOW

            if band is Band.HIGH:
                # Cheap suspicion excludes this answer and queues a verifier;
                # it never writes trust or quarantines by itself.
                excluded.append(doc.doc_id)
                self.queue.enqueue(query, query_id, doc, pool, reason="HIGH band",
                                   timestamp=now)
                continue

            # Every MONITORED passage is re-verified whenever it is used,
            # even if the cheap scorer produced LOW for this particular query.
            if status_before is TrustStatus.MONITORED and band is Band.LOW:
                band = Band.MEDIUM

            if band is Band.MEDIUM:
                if verifier is None:
                    # Fail closed: the contract requires verification before a
                    # MEDIUM/MONITORED passage is admitted to generation.
                    excluded.append(doc.doc_id)
                    unavailable.append(doc.doc_id)
                    continue

                if isinstance(verifier, CorroborationVerifier):
                    result = verifier.verify(
                        query, query_id, doc, pool,
                        same_burst=self.ledger.same_burst,
                        source_influence=lambda src: self.ledger.influence_factor(src, now),
                    )
                else:
                    result = verifier.verify(query, query_id, doc, pool)
                verified.append(result)

                # A result can update trust only if the verifier says this
                # passage influenced the answer and reached SUPPORT/REFUTE.
                if (result.influential and result.outcome in
                        (VerificationOutcome.SUPPORT, VerificationOutcome.REFUTE)):
                    before = self.ledger.get_status(doc.doc_id)
                    strength = (result.support_mass
                                if result.outcome is VerificationOutcome.SUPPORT
                                else result.refute_mass)
                    self.ledger.record_observation(
                        doc.doc_id, doc.source_id, doc.family_id,
                        outcome=result.outcome.value, timestamp=now, strength=strength,
                    )
                    after = self.ledger.get_status(doc.doc_id)
                else:
                    before = after = status_before

                if result.influential and result.outcome is VerificationOutcome.REFUTE:
                    excluded.append(doc.doc_id)
                    if (after in (TrustStatus.QUARANTINED, TrustStatus.REJECTED)
                            and before not in (TrustStatus.QUARANTINED, TrustStatus.REJECTED)):
                        newly_quarantined.append(doc.doc_id)
                else:
                    # SUPPORT and NEUTRAL remain usable; NEUTRAL leaves trust
                    # unchanged and the document remains MONITORED.
                    context.append(doc.doc_id)
                continue

            context.append(doc.doc_id)  # TRUSTED + LOW band

        # Trigger quarantine callback for newly quarantined passages
        if newly_quarantined and self._on_quarantine is not None:
            try:
                self._on_quarantine(newly_quarantined,
                                    reason="refuted by corroboration-gated verifier")
            except Exception:
                logger.warning("on_quarantine callback failed for %s", newly_quarantined,
                               exc_info=True)

        notes: Dict[str, Any] = {
            "policy": "TrustPolicy",
            "newly_quarantined": newly_quarantined,
            "n_verified": len(verified),
            "n_excluded": len(excluded),
            "verification_unavailable": unavailable,
            "n_verification_unavailable": len(unavailable),
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

        Called between queries (or by the queue's background worker) with the
        same verifier the policy uses.  Every outcome is written to the ledger
        exactly like an inline verification, so queued passages can be
        quarantined and trigger remediation later.
        """
        if self.queue is None:
            return []
        ts = now if now is not None else time.time()
        if isinstance(verifier, CorroborationVerifier):
            return self.queue.drain(
                verifier, self.ledger, now=ts, max_items=max_items,
                same_burst=self.ledger.same_burst,
                source_influence=lambda src: self.ledger.influence_factor(src, ts),
            )
        return self.queue.drain(verifier, self.ledger, now=ts,
                                max_items=max_items)
