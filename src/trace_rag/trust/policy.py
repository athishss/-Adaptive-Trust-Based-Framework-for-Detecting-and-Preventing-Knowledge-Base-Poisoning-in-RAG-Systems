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
        self.queue = queue

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
        now = time.time()
        # Trust decay is scheduled per query (plan Section 4.6), and decide() is
        # called exactly once per query by the pipeline.
        self.ledger.begin_query(now)

        context: List[str] = []
        excluded: List[str] = []
        verified: List[VerificationResult] = []
        newly_quarantined: List[str] = []

        for doc in documents:
            assessment = by_id.get(doc.doc_id)
            band = assessment.band if assessment else Band.LOW

            if band is Band.HIGH:
                # Exclude now; verify off the latency path (plan Section 4.4).
                excluded.append(doc.doc_id)
                self.ledger.record_high_band(
                    doc.doc_id, doc.source_id, doc.family_id, timestamp=now,
                )
                if self.queue is not None:
                    self.queue.enqueue(query, query_id, doc, pool,
                                       reason="HIGH band")
                continue

            if band is Band.MEDIUM and verifier is not None:
                # Verify against independent corroboration (using full pool).
                # Only our own verifier accepts the ledger context; any other
                # Verifier implementation keeps the protocol signature.
                if isinstance(verifier, CorroborationVerifier):
                    result = verifier.verify(
                        query, query_id, doc, pool,
                        same_burst=self.ledger.same_burst,
                        source_influence=lambda src: self.ledger.influence_factor(
                            src, now),
                    )
                else:
                    result = verifier.verify(query, query_id, doc, pool)
                verified.append(result)

                # Update trust based on outcome
                self.ledger.record_observation(
                    doc.doc_id, doc.source_id, doc.family_id,
                    outcome=result.outcome.value, timestamp=now,
                )

                if result.outcome is VerificationOutcome.REFUTE:
                    excluded.append(doc.doc_id)
                    # Check if the doc was quarantined by the state machine
                    status = self.ledger.get_status(doc.doc_id)
                    if status.value in ("QUARANTINED", "REJECTED"):
                        newly_quarantined.append(doc.doc_id)
                    continue

                # SUPPORT or NEUTRAL → allow
                context.append(doc.doc_id)
                continue

            if band is Band.MEDIUM and verifier is None:
                # No verifier available → benefit of the doubt
                context.append(doc.doc_id)
                continue

            # LOW band → use
            context.append(doc.doc_id)

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
