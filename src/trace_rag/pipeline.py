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
                        NullTrustProvider, Policy, PolicyDecision, SecurityAssessment, TrustProvider, Verifier)
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
        self._owned_trust_provider = None
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

        owned_ledger = None
        auto_queue = None
        auto_policy = False
        if config.trust.enabled:
            from .trust.ledger import TrustLedger
            from .trust.policy import TrustPolicy
            from .trust.queue import VerificationQueue
            from .trust.verifier import CorroborationVerifier

            trust_provider = kwargs.get("trust_provider")
            if trust_provider is None:
                owned_ledger = TrustLedger(
                    config.path(config.storage.trust_db), store=store,
                    config=config.trust.to_ledger_config(),
                )
                trust_provider = owned_ledger
                kwargs["trust_provider"] = trust_provider
            if kwargs.get("llm") is None:
                kwargs["llm"] = build_llm(config.generation)
            if kwargs.get("verifier") is None:
                kwargs["verifier"] = CorroborationVerifier(
                    llm=kwargs["llm"], **config.trust.verifier_kwargs(),
                )
            auto_queue = VerificationQueue(max_size=config.trust.queue_max_size)
            if kwargs.get("policy") is None and isinstance(trust_provider, TrustLedger):
                # The callback needs the constructed pipeline, so install the
                # policy immediately after __init__ below.
                auto_policy = True
                kwargs["policy"] = DefaultPolicy()

        pipeline = cls(config=config, store=store, index=index, embedder=embedder,
                       scorer=scorer, **kwargs)
        if config.trust.enabled and auto_policy and isinstance(pipeline.trust_provider, TrustLedger):
            pipeline.policy = TrustPolicy(
                    ledger=pipeline.trust_provider,
                    on_quarantine=pipeline.on_quarantine,
                    queue=auto_queue,
                )
        pipeline._owned_trust_provider = owned_ledger
        return pipeline

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
                                          self.verifier, pool=retrieval.pool, now=now)
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
                     train_sample: int = 100_000, progress_every: int = 5_000,
                     limit: Optional[int] = None, save_every: int = 0) -> int:
        """Embed and index chunks from the provenance store.

        Streams from SQLite in batches, so a 2.68M-passage corpus never sits in
        memory.  **Resumable**: passages already in the index are skipped, and
        ``save_every`` checkpoints the index to disk, so a crash, a power cut or
        a Ctrl+C costs at most that many passages instead of the whole run.
        ``limit`` stops after that many newly indexed passages, which is how you
        index a corpus in sittings.
        """
        if chunk_ids is not None:
            records = sorted(self.store.get_chunks(chunk_ids).values(), key=lambda r: r.chunk_id)
            if not records:
                return 0
            self._train_index_if_needed(iter(records), train_sample)
            return self._add_records(iter(records), batch_size, progress_every, limit, save_every)

        pending = self.store.counts().get("chunks", 0)
        logger.info("indexing %s passages with %s on %s (this is the slow step)",
                    f"{pending:,}", getattr(self.embedder, "model_name",
                                            getattr(self.embedder, "name", "embedder")),
                    getattr(self.embedder, "device", "cpu"))
        already = len(self.index)
        if already:
            logger.info("%s passages are already indexed; they will be skipped", f"{already:,}")
        needs_training = hasattr(self.index, "is_trained") and not self.index.is_trained
        if needs_training:
            self._train_index_if_needed(self.store.iter_chunks(batch_size=1000), train_sample)
        return self._add_records(self.store.iter_chunks(batch_size=1000), batch_size,
                                 progress_every, limit, save_every)

    def _train_index_if_needed(self, records, train_sample: int) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self.index, "is_trained") or self.index.is_trained:
            return
        texts: List[str] = []
        for record in records:
            texts.append(record.text)
            if len(texts) >= train_sample:
                break
        if not texts:
            return
        logger.info("training index on %d sampled passages", len(texts))
        vectors = self.embedder.encode_documents(texts, batch_size=self.config.embedding.batch_size)
        self.index.train(vectors)

    def _add_records(self, records, batch_size: int, progress_every: int,  # type: ignore[no-untyped-def]
                     limit: Optional[int] = None, save_every: int = 0) -> int:
        total = 0
        milestone = 0
        checkpoint = 0
        started = time.perf_counter()
        buffer_ids: List[str] = []
        buffer_texts: List[str] = []

        def flush() -> None:
            nonlocal total, milestone, checkpoint
            if not buffer_ids:
                return
            vectors = self.embedder.encode_documents(
                buffer_texts, batch_size=self.config.embedding.batch_size)
            self.index.add(list(buffer_ids), vectors)
            total += len(buffer_ids)
            buffer_ids.clear()
            buffer_texts.clear()
            # Report on every completed block, so a long run never looks stuck.
            if progress_every and total // progress_every > milestone:
                milestone = total // progress_every
                elapsed = time.perf_counter() - started
                rate = total / elapsed if elapsed > 0 else 0.0
                logger.info("indexed %s passages (%.0f/s, %.1f min elapsed)",
                            f"{total:,}", rate, elapsed / 60.0)
            if save_every and total // save_every > checkpoint:
                checkpoint = total // save_every
                path = self.save_index()
                logger.info("checkpoint saved to %s (%s passages)", path, f"{total:,}")

        for record in records:
            if record.chunk_id in self.index:          # resume: already embedded
                continue
            buffer_ids.append(record.chunk_id)
            buffer_texts.append(record.text)
            if len(buffer_ids) >= batch_size:
                flush()
                if limit is not None and total >= limit:
                    logger.info("stopping at the requested limit of %s passages", f"{limit:,}")
                    break
        flush()
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
        if self._owned_trust_provider is not None:
            close = getattr(self._owned_trust_provider, "close", None)
            if close is not None:
                close()
            self._owned_trust_provider = None
        self.store.close()
        self.answer_log.close()
