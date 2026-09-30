"""Trust-weighted retrieval (plan Section 4.6).

    score(d) = similarity(q, d) * max(t_eff(d), floor) ** lambda

over a candidate pool of K' passages, re-ranked down to top-k.  Quarantined and
rejected documents are removed from the pool entirely.

The trust floor exists for cold-start fairness: a brand-new *honest* source
must still be reachable, otherwise the defence silently censors new content.
Person C measures that effect (clean recall for new sources).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RetrievalConfig
from ..contracts import NullTrustProvider, RetrievedDocument, TrustProvider, TrustSnapshot
from ..ingestion.provenance import ChunkRecord, ProvenanceStore
from ..utils.timing import Stopwatch


@dataclass
class RetrievalOutcome:
    """Top-k plus the full candidate pool.

    Person B's verifier needs the pool: corroboration is computed against
    passages from *independent* sources, which usually sit outside the top-k.
    """

    query: str
    query_id: str
    documents: Tuple[RetrievedDocument, ...]
    pool: Tuple[RetrievedDocument, ...]
    query_vector: np.ndarray
    latency_ms: float = 0.0
    n_blocked: int = 0

    def doc_ids(self) -> Tuple[str, ...]:
        return tuple(d.doc_id for d in self.documents)

    def by_id(self) -> Dict[str, RetrievedDocument]:
        return {d.doc_id: d for d in self.pool}


class TrustWeightedRetriever:
    def __init__(self, store: ProvenanceStore, index, embedder,  # type: ignore[no-untyped-def]
                 config: Optional[RetrievalConfig] = None,
                 trust_provider: Optional[TrustProvider] = None) -> None:
        self.store = store
        self.index = index
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.trust = trust_provider or NullTrustProvider()

    # ------------------------------------------------------------------ core
    def retrieve(self, query: str, query_id: str = "q0", k: Optional[int] = None,
                 pool_size: Optional[int] = None) -> RetrievalOutcome:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = int(k or self.config.top_k)
        pool_size = int(pool_size or self.config.candidate_pool)
        if pool_size < k:
            pool_size = k

        with Stopwatch() as watch:
            query_vector = self.embedder.encode_queries([query])[0]
            blocked = self.trust.blocked_doc_ids() if self.config.exclude_blocked else set()
            hits = self.index.search(query_vector[None, :], pool_size, exclude=blocked)[0]

            chunk_ids = [chunk_id for chunk_id, _ in hits]
            records = self.store.get_chunks(chunk_ids)
            trust_map = self.trust.get_trust(chunk_ids)

            pool: List[RetrievedDocument] = []
            for rank, (chunk_id, similarity) in enumerate(hits):
                record = records.get(chunk_id)
                if record is None:                     # index/store drift: skip, never crash
                    continue
                snapshot = trust_map.get(chunk_id) or TrustSnapshot.neutral(
                    chunk_id, record.source_id, record.family_id)
                pool.append(self._make_document(record, similarity, rank, snapshot))

            ranked = sorted(pool, key=lambda d: (-self._score(d), d.doc_id))
            top = tuple(
                RetrievedDocument(
                    doc_id=d.doc_id, chunk_id=d.chunk_id, text=d.text, similarity=d.similarity,
                    rank=i, source_id=d.source_id, family_id=d.family_id, trust=d.trust,
                    metadata={**d.metadata, "trust_score": self._score(d)},
                )
                for i, d in enumerate(ranked[:k])
            )
        return RetrievalOutcome(
            query=query, query_id=query_id, documents=top, pool=tuple(pool),
            query_vector=query_vector, latency_ms=watch.elapsed_ms, n_blocked=len(blocked),
        )

    def retrieve_batch(self, queries: Sequence[str], query_ids: Optional[Sequence[str]] = None,
                       k: Optional[int] = None, pool_size: Optional[int] = None
                       ) -> List[RetrievalOutcome]:
        ids = list(query_ids or [f"q{i}" for i in range(len(queries))])
        if len(ids) != len(queries):
            raise ValueError("query_ids length must match queries")
        return [self.retrieve(q, qid, k, pool_size) for q, qid in zip(queries, ids)]

    # ---------------------------------------------------------------- helpers
    def _score(self, document: RetrievedDocument) -> float:
        trust = max(float(document.trust.t_eff), self.config.trust_floor)
        return float(document.similarity) * (trust ** self.config.trust_lambda)

    @staticmethod
    def _make_document(record: ChunkRecord, similarity: float, rank: int,
                       snapshot: TrustSnapshot) -> RetrievedDocument:
        return RetrievedDocument(
            doc_id=record.chunk_id,            # chunk-level ids are what trust tracks
            chunk_id=record.chunk_id,
            text=record.text,
            similarity=float(similarity),
            rank=int(rank),
            source_id=record.source_id,
            family_id=record.family_id,
            trust=snapshot,
            metadata={
                "parent_doc_id": record.doc_id,
                "ingested_at": record.ingested_at,
                "n_words": record.n_words,
                "sha256": record.sha256,
            },
        )
