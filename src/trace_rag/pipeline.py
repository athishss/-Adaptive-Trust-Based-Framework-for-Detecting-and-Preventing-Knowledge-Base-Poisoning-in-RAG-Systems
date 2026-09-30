"""PersonAPipeline: the assembled Person A system, with plug points for B and C.

Flow for one query (plan Section 4.2):

    retrieve (trust-weighted)  ->  cheap signals  ->  suspicion scorer
        ->  [Person B policy + verifier]  ->  grounded generation
        ->  answer-provenance log

Person B plugs in by passing ``trust_provider``, ``verifier`` and ``policy``.
Person C drives the whole thing through :meth:`answer` and reads
``AnswerRecord`` / ``SecurityAssessment`` objects.  Nothing in this class reads
attack labels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Config
from .contracts import (Action, AnswerRecord, Band, DefaultPolicy, FeatureSnapshot, LLM,
                        NullTrustProvider, Policy, PolicyDecision, RetrievedDocument,
                        SecurityAssessment, TrustProvider, Verifier)
from .detection.scorer import HeuristicScorer, SuspicionScorer
from .detection.signals import SignalComputer
from .embeddings import build_embedder
from .generation.grounded import GenerationOutcome, GroundedGenerator
from .generation.llm import build_llm
from .index import build_index, load_index
from .ingestion.provenance import ProvenanceStore
from .provenance.answer_log import AnswerLog
from .provenance.remediation import RemediationService
from .retrieval.retriever import RetrievalOutcome, TrustWeightedRetriever
from .utils.logging import get_logger
from .utils.timing import Stopwatch

logger = get_logger(__name__)

_BAND_ACTION = {Band.LOW: Action.USE, Band.MEDIUM: Action.VERIFY_THEN_USE, Band.HIGH: Action.EXCLUDE}


@dataclass
class QueryResult:
    """Everything one query produced.  This is what Person C collects."""

    query: str
    query_id: str
    retrieval: RetrievalOutcome
    assessments: Tuple[SecurityAssessment, ...]
    decision: PolicyDecision
    generation: GenerationOutcome
    answer_id: Optional[int]
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.generation.record.answer

    @property
    def record(self) -> AnswerRecord:
        return self.generation.record

    @property
    def abstained(self) -> bool:
        return self.generation.record.abstained

    @property
    def llm_calls(self) -> int:
        verification_calls = sum(v.llm_calls for v in self.decision.verified)
        return int(self.generation.record.llm_calls + verification_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.record.to_dict(),
            "retrieved": [d.to_dict() for d in self.retrieval.documents],
            "assessments": [a.to_dict() for a in self.assessments],
            "decision": self.decision.to_dict(),
            "answer_id": self.answer_id,
            "llm_calls_total": self.llm_calls,
            "timings_ms": dict(self.timings_ms),
        }


class PersonAPipeline:
    def __init__(self, config: Config, store: ProvenanceStore, index, embedder,  # type: ignore[no-untyped-def]
                 llm: Optional[LLM] = None, scorer: Optional[Any] = None,
                 trust_provider: Optional[TrustProvider] = None,
                 verifier: Optional[Verifier] = None, policy: Optional[Policy] = None,
                 answer_log: Optional[AnswerLog] = None,
                 citation_checker: Optional[Callable[[str, str], Tuple[bool, float]]] = None) -> None:
        self.config = config
        self.store = store
        self.index = index
        self.embedder = embedder
        self.trust_provider = trust_provider or NullTrustProvider()
        self.verifier = verifier
        self.policy = policy or DefaultPolicy()
        self.scorer = scorer or HeuristicScorer(config.scorer.theta_low, config.scorer.theta_high)
        self.retriever = TrustWeightedRetriever(store, index, embedder, config.retrieval,
                                                self.trust_provider)
        self.signals = SignalComputer(store, index, config.signals)
        self.generator = GroundedGenerator(llm or build_llm(config.generation), config.generation,
                                           citation_checker=citation_checker)
        self.answer_log = answer_log or AnswerLog(config.path(config.storage.answer_db))
        self.remediation = RemediationService(self.answer_log)

    # --------------------------------------------------------------- factory
    @classmethod
    def from_config(cls, config: Config, load_existing_index: bool = True, **kwargs: Any
                    ) -> "PersonAPipeline":
        store = ProvenanceStore(config.path(config.storage.provenance_db))
        embedder = build_embedder(config.embedding)
        index_path = config.path(config.storage.index_path)
        index = None
        if load_existing_index:
            for suffix in (".npz", ".faiss"):
                if Path(str(index_path) + suffix).exists():
                    index = load_index(config.index, str(index_path))
                    break
        if index is None:
            index = build_index(config.index, embedder.dim)
        scorer = kwargs.pop("scorer", None)
        if scorer is None and config.scorer.model_path and Path(config.scorer.model_path).exists():
            scorer = SuspicionScorer.load(config.scorer.model_path)
        return cls(config=config, store=store, index=index, embedder=embedder, scorer=scorer, **kwargs)

    # ------------------------------------------------------------------ core
    def retrieve(self, query: str, query_id: str = "q0") -> RetrievalOutcome:
        return self.retriever.retrieve(query, query_id)

    def assess(self, outcome: RetrievalOutcome, now: Optional[float] = None
               ) -> Tuple[SecurityAssessment, ...]:
        """Cheap signals + calibrated suspicion for each retrieved passage."""
        snapshots: Dict[str, FeatureSnapshot] = self.signals.compute(
            outcome.query, outcome.documents, outcome.pool, now=now)
        assessments: List[SecurityAssessment] = []
        for doc in outcome.documents:
            snapshot = snapshots.get(doc.doc_id)
            if snapshot is None:
                continue
            suspicion = float(self.scorer.score_snapshot(snapshot))
            band = self.scorer.band(suspicion)
            assessments.append(SecurityAssessment(
                doc_id=doc.doc_id, chunk_id=doc.chunk_id, query_id=outcome.query_id,
                signals=snapshot.signals, suspicion=suspicion, band=band,
                action=_BAND_ACTION[band], feature_snapshot=snapshot,
                scorer_version=getattr(self.scorer, "version", "unknown"),
            ))
        return tuple(assessments)

    def answer(self, query: str, query_id: Optional[str] = None, step: int = 0,
               now: Optional[float] = None, log: bool = True) -> QueryResult:
        """Run the full pipeline for one query."""
        query_id = query_id or f"q{int(time.time() * 1000)}"
        timings: Dict[str, float] = {}

        with Stopwatch() as watch:
            retrieval = self.retriever.retrieve(query, query_id)
        timings["retrieval"] = watch.elapsed_ms

        with Stopwatch() as watch:
            assessments = self.assess(retrieval, now=now)
        timings["signals_and_scoring"] = watch.elapsed_ms

        with Stopwatch() as watch:
            decision = self.policy.decide(query, query_id, retrieval.documents, assessments,
                                          self.verifier)
        timings["policy"] = watch.elapsed_ms

        by_id = {d.doc_id: d for d in retrieval.documents}
        context = [by_id[doc_id] for doc_id in decision.context_doc_ids if doc_id in by_id]

        with Stopwatch() as watch:
            generation = self.generator.generate(query, query_id, context,
                                                 excluded_doc_ids=decision.excluded_doc_ids, step=step)
        timings["generation"] = watch.elapsed_ms
        timings["total"] = sum(timings.values())

        answer_id = None
        if log:
            answer_id = self.answer_log.record(generation.record,
                                               context_doc_ids=decision.context_doc_ids)
        return QueryResult(query=query, query_id=query_id, retrieval=retrieval,
                           assessments=assessments, decision=decision, generation=generation,
                           answer_id=answer_id, timings_ms=timings)

    def answer_stream(self, queries: Sequence[Mapping[str, Any]]) -> List[QueryResult]:
        """Run a query stream (Person C's ``stream.jsonl`` rows)."""
        results: List[QueryResult] = []
        for i, row in enumerate(queries):
            results.append(self.answer(
                query=str(row["text"]), query_id=str(row.get("qid", f"q{i}")),
                step=int(row.get("t", i)),
            ))
        return results

    # ------------------------------------------------------------ maintenance
    def index_chunks(self, chunk_ids: Optional[Sequence[str]] = None, batch_size: int = 256,
                     train_sample: int = 100_000) -> int:
        """Embed and index chunks from the provenance store.

        Passing ``chunk_ids`` indexes just those (used when Person C injects a
        new batch mid-stream); otherwise the whole store is indexed.
        """
        records = (list(self.store.get_chunks(chunk_ids).values()) if chunk_ids
                   else list(self.store.iter_chunks()))
        records.sort(key=lambda r: r.chunk_id)
        if not records:
            return 0
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            sample = records[:train_sample]
            vectors = self.embedder.encode_documents([r.text for r in sample],
                                                     batch_size=self.config.embedding.batch_size)
            self.index.train(vectors)
        total = 0
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            vectors = self.embedder.encode_documents([r.text for r in batch],
                                                     batch_size=self.config.embedding.batch_size)
            self.index.add([r.chunk_id for r in batch], vectors)
            total += len(batch)
        logger.info("indexed %d chunks (index size %d)", total, len(self.index))
        return total

    def save_index(self) -> str:
        path = str(self.config.path(self.config.storage.index_path))
        self.index.save(path)
        return path

    def on_quarantine(self, doc_ids: Sequence[str], reason: str = "quarantined by trust ledger"):
        """Hook for Person B's state machine; returns a RemediationReport."""
        return self.remediation.on_quarantine(doc_ids, reason)

    def stats(self) -> Dict[str, Any]:
        return {
            "store": self.store.counts(),
            "index_size": len(self.index),
            "answers": self.answer_log.stats(),
            "scorer": getattr(self.scorer, "version", "unknown"),
            "embedder": getattr(self.embedder, "name", "unknown"),
            "llm": getattr(self.generator.llm, "name", "unknown"),
            "trust_provider": type(self.trust_provider).__name__,
            "verifier": type(self.verifier).__name__ if self.verifier else None,
            "policy": type(self.policy).__name__,
        }

    def close(self) -> None:
        self.store.close()
        self.answer_log.close()
