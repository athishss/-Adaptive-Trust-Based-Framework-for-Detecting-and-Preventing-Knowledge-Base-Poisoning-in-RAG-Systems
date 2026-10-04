"""Asynchronous verification queue (plan Section 4.4, B4).

A HIGH-band passage is excluded from the answer immediately, because the
latency budget cannot wait for verification, and queued instead.  The queue is
drained between queries -- :meth:`VerificationQueue.drain` -- or handed to a
daemon worker with :meth:`VerificationQueue.start_background`, so verification
happens off the latency path.

Draining writes every outcome to the ledger exactly like an inline
verification, so a passage queued while answering query 12 can be quarantined
before query 13 and have its earlier answers remediated.

Entirely person B's component: it only needs ``Verifier`` and ``TrustLedger``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ..contracts import RetrievedDocument, Verifier, VerificationResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PendingVerification:
    """One queued passage waiting for off-path verification."""

    query: str
    query_id: str
    doc: RetrievedDocument
    pool: Sequence[RetrievedDocument]
    reason: str = "HIGH band"
    enqueued_at: float = 0.0


class VerificationQueue:
    """Bounded FIFO of passages awaiting verification.

    One entry per ``doc_id``: a document seen again on a later query replaces
    its earlier entry (the newest pool is the one worth corroborating against).
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = int(max_size)
        self._items: "deque[PendingVerification]" = deque()
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.dropped: int = 0
        self.processed: int = 0

    # ------------------------------------------------------------------
    #  Producer side (the policy)
    # ------------------------------------------------------------------

    def enqueue(self, query: str, query_id: str, doc: RetrievedDocument,
                pool: Sequence[RetrievedDocument], reason: str = "HIGH band",
                timestamp: Optional[float] = None) -> bool:
        """Queue *doc* for verification.  Returns False if it was already queued."""
        item = PendingVerification(
            query=query, query_id=query_id,
            doc=doc, pool=tuple(pool), reason=reason,
            enqueued_at=timestamp if timestamp is not None else time.time(),
        )
        with self._lock:
            for existing in self._items:
                if existing.doc.doc_id == doc.doc_id:
                    existing.query = item.query
                    existing.query_id = item.query_id
                    existing.pool = item.pool
                    existing.reason = item.reason
                    existing.enqueued_at = item.enqueued_at
                    return False
            self._items.append(item)
            while len(self._items) > self.max_size:
                self._items.popleft()
                self.dropped += 1
                logger.warning("verification queue full (%d): dropped oldest item",
                               self.max_size)
        return True

    def pending(self) -> List[PendingVerification]:
        """Snapshot of the queued items, oldest first."""
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    # ------------------------------------------------------------------
    #  Consumer side (the runner)
    # ------------------------------------------------------------------

    def drain(self, verifier: Optional[Verifier], ledger: Any = None,
              now: Optional[float] = None,
              max_items: Optional[int] = None,
              **verify_kwargs: Any) -> List[VerificationResult]:
        """Verify queued passages and record their outcomes.

        Called between queries, so it is deliberately synchronous and
        deterministic: it pops at most *max_items* entries (all of them by
        default), verifies each, and writes SUPPORT/REFUTE/NEUTRAL into the
        ledger.  ``verify_kwargs`` are forwarded to the verifier (the policy
        passes the ledger's ``same_burst`` / ``source_influence``).
        """
        if verifier is None:
            return []
        ts = now if now is not None else time.time()
        results: List[VerificationResult] = []
        processed = 0
        while max_items is None or processed < max_items:
            with self._lock:
                if not self._items:
                    break
                item = self._items.popleft()
            processed += 1
            try:
                result = verifier.verify(item.query, item.query_id,
                                         item.doc, item.pool, **verify_kwargs)
            except Exception:
                # One bad case must not kill the worker; make the failure loud.
                logger.warning("queued verification failed for %s",
                               item.doc.doc_id, exc_info=True)
                continue
            if ledger is not None:
                ledger.record_observation(
                    item.doc.doc_id, item.doc.source_id, item.doc.family_id,
                    outcome=result.outcome.value, timestamp=ts,
                )
            results.append(result)
            self.processed += 1
        return results

    # ------------------------------------------------------------------
    #  Background worker
    # ------------------------------------------------------------------

    def start_background(self, verifier: Verifier, ledger: Any = None,
                         interval: float = 0.5, **verify_kwargs: Any) -> None:
        """Run :meth:`drain` in a daemon thread until ``stop_background``."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.drain(verifier, ledger, **verify_kwargs)
                except Exception:  # pragma: no cover - defensive
                    logger.warning("verification worker iteration failed",
                                   exc_info=True)
                self._stop.wait(interval)

        self._worker = threading.Thread(target=_loop, name="trace-rag-verifier",
                                        daemon=True)
        self._worker.start()

    def stop_background(self, timeout: float = 5.0) -> None:
        """Stop the worker thread and wait for it to finish."""
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():  # pragma: no cover - only on a stuck drain
                logger.warning("verification worker did not stop within %.1fs",
                               timeout)
        self._worker = None
