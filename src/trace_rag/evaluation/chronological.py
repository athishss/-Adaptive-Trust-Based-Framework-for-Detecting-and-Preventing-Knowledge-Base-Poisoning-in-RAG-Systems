"""Chronological evaluation harness (Person C).

The runner executes the real system, not a simulation of it:

1. ingest events at their stream step (with that step's timestamp);
2. index the new chunks **before** answering queries at that step;
3. answer queries through the pipeline's defence (retrieval, signals, policy,
   verification, generation);
4. drain deferred verification;
5. record per-query ground truth: was a poison passage retrieved? was it
   allowed into the generation context? was it cited? was it quarantined?
6. compute document-level and query-level metrics on one step timeline;
7. write machine-readable results.

Ground truth is *event-level*: a poison passage counts as successful only if it
was actually ingested before the target query and then either entered the
generation context or was cited.  An attack object that merely exists, a query
that abstained for an unrelated reason, or a passage that was never retrieved is
recorded explicitly and is not counted as a block.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..attacks.stream import (StreamIngestion, StreamPlan, StreamQuery,
                              as_ingestion_row, as_query_row, build_stream_plan)
from ..contracts import VerificationResult
from ..detection.training import LabelledRow
from ..ingestion.pipeline import Ingestor
from ..pipeline import PersonAPipeline
from ..trust.policy import TrustPolicy
from ..trust.verifier import CorroborationVerifier
from .metrics import (gold_answer_present, abstention_rate, answer_level_asr_fields, attack_answer_success,
                      clean_utility, detection_delays, detection_metrics, exposure_windows,
                      latency_stats, mean_llm_calls, poison_citation_rate,
                      poison_exposure_rate, summarise_optional)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StreamExecutionError(RuntimeError):
    """The stream could not be executed safely (bad schedule, missing index, ...)."""


@dataclass
class QueryStepRecord:
    step: int
    query_id: str
    query: str
    is_target: bool
    answer: str
    abstained: bool
    abstain_reason: Optional[str]
    retrieved_ids: Tuple[str, ...]
    context_ids: Tuple[str, ...]
    excluded_ids: Tuple[str, ...]
    cited_ids: Tuple[str, ...]
    poison_targeted: Tuple[str, ...]
    poison_retrieved: Tuple[str, ...]
    poison_in_context: Tuple[str, ...]
    poison_cited: Tuple[str, ...]
    bands: Mapping[str, int]
    verified: Tuple[Tuple[str, str], ...]
    drained: Tuple[Tuple[str, str], ...]
    newly_quarantined: Tuple[str, ...]
    llm_calls: int
    latency_ms: float
    answer_id: Optional[int]
    answer_attack_success: Optional[bool] = None
    answer_gold_correct: Optional[bool] = None
    answer_attack_status: str = "NOT MEASURED"
    attack_answer_criterion: Optional[str] = None
    result: Any = None

    @property
    def poison_exposed(self) -> bool:
        """Whether poison reached the generation context.

        Citation alone is reported separately; it is not context exposure.
        """
        return bool(self.poison_in_context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "query_id": self.query_id,
            "query": self.query,
            "is_target": bool(self.is_target),
            "answer": self.answer,
            "abstained": bool(self.abstained),
            "abstain_reason": self.abstain_reason,
            "retrieved_ids": list(self.retrieved_ids),
            "context_ids": list(self.context_ids),
            "excluded_ids": list(self.excluded_ids),
            "cited_ids": list(self.cited_ids),
            "poison_targeted": list(self.poison_targeted),
            "poison_retrieved": list(self.poison_retrieved),
            "poison_in_context": list(self.poison_in_context),
            "poison_cited": list(self.poison_cited),
            "poison_exposed": self.poison_exposed,
            "poison_cited_any": bool(self.poison_cited),
            "bands": dict(self.bands),
            "verified": [{"doc_id": d, "outcome": o} for d, o in self.verified],
            "verification_details": [
                {
                    "doc_id": result.doc_id,
                    "outcome": _status(result.outcome),
                    "influential": bool(result.influential),
                    "support_mass": float(result.support_mass),
                    "refute_mass": float(result.refute_mass),
                    "evidence_doc_ids": list(result.evidence_doc_ids),
                    "error": result.error,
                }
                for result in (
                    list(self.result.decision.verified)
                    if self.result is not None else []
                )
            ],
            "drained": [{"doc_id": d, "outcome": o} for d, o in self.drained],
            "newly_quarantined": list(self.newly_quarantined),
            "llm_calls": int(self.llm_calls),
            "latency_ms": float(self.latency_ms),
            "answer_id": self.answer_id,
            "answer_attack_success": self.answer_attack_success,
            "answer_gold_correct": self.answer_gold_correct,
            "answer_attack_status": self.answer_attack_status,
            "attack_answer_criterion": self.attack_answer_criterion,
        }


@dataclass
class StreamRun:
    plan: StreamPlan
    records: List[QueryStepRecord]
    metrics: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    injection_steps: Dict[str, int] = field(default_factory=dict)
    poison_doc_ids: Set[str] = field(default_factory=set)
    training_rows: List[LabelledRow] = field(default_factory=list)

    @property
    def answers(self) -> List[Any]:
        return [record.result for record in self.records]

    @property
    def stream_end(self) -> int:
        steps = [record.step for record in self.records]
        steps.extend(event.step for event in self.plan.ingestions)
        return max(steps) if steps else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "plan": self.plan.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }

    def write(self, output_dir: str | Path) -> Dict[str, str]:
        """Write JSON, JSONL and a human summary.  Returns the paths."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics_path = out / "metrics.json"
        records_path = out / "stream.jsonl"
        plan_path = out / "plan.json"
        config_path = out / "run_config.json"
        summary_path = out / "summary.md"

        metrics_path.write_text(json.dumps(self.metrics, indent=2, sort_keys=True),
                                encoding="utf-8")
        records_path.write_text(
            "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in self.records),
            encoding="utf-8")
        plan_path.write_text(json.dumps(self.plan.to_dict(), indent=2), encoding="utf-8")
        config_path.write_text(json.dumps(self.config, indent=2, sort_keys=True),
                               encoding="utf-8")
        summary_path.write_text(_summary_markdown(self), encoding="utf-8")
        if self.training_rows:
            payload = "\n".join(json.dumps({
                "query_id": row.query_id, "doc_id": row.doc_id,
                "label": row.label, "attack_family": row.attack_family,
                "captured_at": row.captured_at, "query_time": row.query_time,
                "features": row.features.to_dict(),
            }, sort_keys=True) for row in self.training_rows)
            (out / "training_rows.jsonl").write_text(payload, encoding="utf-8")
        return {"metrics": str(metrics_path), "records": str(records_path),
                "plan": str(plan_path), "config": str(config_path),
                "summary": str(summary_path)}


def _status(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _ingest_step_events(ingestor: Ingestor, events: Sequence[Mapping[str, Any]],
                        timestamp: float) -> Dict[str, Tuple[str, ...]]:
    """Ingest one step's documents and return ``doc_id -> chunk_ids``."""
    produced: Dict[str, Tuple[str, ...]] = {}
    for event in events:
        records = ingestor.ingest_text(
            doc_id=event["doc_id"], text=event["text"], source_id=event["source_id"],
            title=event.get("title", ""), origin=event.get("origin", "person_c"),
            ingested_at=timestamp, passage_mode=True,
            metadata={
                "is_poison": bool(event.get("is_poison", False)),
                "attack_type": event.get("attack_type"),
                "family_id": event.get("family_id"),
                "target_query": event.get("target_query"),
                **(event.get("metadata") or {}),
            },
        )
        if not records:
            logger.warning("ingestion produced no chunks for %s", event["doc_id"])
        produced[event["doc_id"]] = tuple(record.chunk_id for record in records)
    return produced


def run_stream(
    pipeline: PersonAPipeline,
    ingestor: Ingestor,
    events: Sequence[Any],
    queries: Sequence[Any],
    *,
    poison_ids: Optional[Iterable[str]] = None,
    clean_ids: Optional[Iterable[str]] = None,
    target_query_ids: Optional[Iterable[str]] = None,
    answer_ground_truth: Optional[Mapping[str, Mapping[str, Any]]] = None,
    output_dir: Optional[str | Path] = None,
    drain_queue: bool = True,
    log: bool = True,
) -> StreamRun:
    """Execute a chronological stream and return records plus metrics.

    ``events`` / ``queries`` accept the canonical rows *and* the
    :class:`StreamIngestion` / :class:`StreamQuery` dataclasses; both go through
    the shared adapters in ``attacks.stream``.
    """
    canonical_events = [as_ingestion_row(event) for event in events]
    canonical_queries = [as_query_row(query, index=i) for i, query in enumerate(queries)]
    target_ids = set(target_query_ids or ())
    target_texts = {
        str(getattr(query, "target_query", None))
        for query in queries
        if getattr(query, "target_query", None) and getattr(query, "is_target", False)
    }

    stream_queries: List[StreamQuery] = []
    for index, (raw, row) in enumerate(zip(queries, canonical_queries)):
        is_target = (row["query_id"] in target_ids or row["query"] in target_texts
                     or bool(getattr(raw, "is_target", False)))
        stream_queries.append(StreamQuery(
            step=row["step"], query_id=row["query_id"], query=row["query"],
            target_query=row["query"] if is_target else None, is_target=is_target,
        ))

    ingestions = [StreamIngestion(**event) for event in canonical_events]
    plan = build_stream_plan(stream_queries, ingestions)

    clock = pipeline.clock
    events_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for event in canonical_events:
        events_by_step.setdefault(int(event["step"]), []).append(event)
    queries_by_step: Dict[int, List[StreamQuery]] = {}
    for query in stream_queries:
        queries_by_step.setdefault(query.step, []).append(query)

    # doc_id -> chunk ids, and the inverse; the pipeline works in chunk ids.
    doc_to_chunks: Dict[str, Tuple[str, ...]] = {}
    chunk_to_doc: Dict[str, str] = {}
    injection_steps: Dict[str, int] = {}
    poison_ground_truth: Set[str] = set(poison_ids or ())
    poison_by_target_query: Dict[str, Set[str]] = {}
    chunk_attack: Dict[str, str] = {}
    training_rows: List[LabelledRow] = []
    declared_clean: Set[str] = set(clean_ids or ())

    records: List[QueryStepRecord] = []
    answer_ground_truth = answer_ground_truth or {}
    flagged_docs: Set[str] = set()
    first_detection: Dict[str, int] = {}
    first_quarantine: Dict[str, int] = {}
    poison_answers: Set[int] = set()

    for step in sorted(set(events_by_step) | set(queries_by_step)):
        timestamp = clock.timestamp(step)
        clock.set_step(step)
        produced = _ingest_step_events(ingestor, events_by_step.get(step, []), timestamp)
        new_chunk_ids = [chunk_id for chunks in produced.values() for chunk_id in chunks]
        for event in events_by_step.get(step, []):
            chunks = produced.get(event["doc_id"], ())
            doc_to_chunks[event["doc_id"]] = tuple(chunks)
            for chunk_id in chunks:
                chunk_to_doc[chunk_id] = event["doc_id"]
                if event.get("is_poison"):
                    poison_ground_truth.add(chunk_id)
                    injection_steps[chunk_id] = step
                    chunk_attack[chunk_id] = str(event.get("attack_type") or "poison")
                    if event.get("target_query"):
                        poison_by_target_query.setdefault(str(event["target_query"]), set()).add(chunk_id)
        if new_chunk_ids:
            try:
                pipeline.index_chunks(chunk_ids=new_chunk_ids)
            except Exception as exc:  # noqa: BLE001
                raise StreamExecutionError(
                    f"indexing {len(new_chunk_ids)} chunks at step {step} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        for query in queries_by_step.get(step, []):
            result = pipeline.answer(query.query, query_id=query.query_id,
                                     step=query.step, now=timestamp, log=log)
            if drain_queue and getattr(pipeline, "security", None) is None:
                policy = getattr(pipeline, "policy", None)
                if hasattr(policy, "drain_verification_queue"):
                    policy.drain_verification_queue(getattr(pipeline, "verifier", None),
                                                    now=timestamp)

            retrieved = tuple(d.doc_id for d in result.retrieval.documents)
            context = tuple(result.decision.context_doc_ids)
            excluded = tuple(result.decision.excluded_doc_ids)
            cited = tuple(result.record.used_doc_ids)
            targeted = _targeted_poison(query, poison_by_target_query, poison_ground_truth,
                                        injection_steps, step)
            poison_retrieved = tuple(doc for doc in retrieved if doc in targeted)
            poison_context = tuple(doc for doc in context if doc in targeted)
            poison_cited = tuple(doc for doc in cited if doc in targeted)

            band_counts = {band: 0 for band in ("LOW", "MEDIUM", "HIGH")}
            for assessment in result.assessments:
                # Features are snapshotted at this retrieval; labels are used
                # only by the offline trainer, never by the live defence.
                training_rows.append(LabelledRow(
                    query_id=query.query,
                    doc_id=assessment.doc_id,
                    label=int(assessment.doc_id in poison_ground_truth),
                    features=assessment.feature_snapshot,
                    attack_family=chunk_attack.get(assessment.doc_id, "none"),
                    captured_at=timestamp, query_time=timestamp,
                ))
                band_counts[assessment.band.value] = band_counts.get(assessment.band.value, 0) + 1
                if assessment.band.value == "HIGH":
                    # A positive detection: the suspicion scorer escalated it
                    # to the highest band.
                    flagged_docs.add(assessment.doc_id)
                    first_detection.setdefault(assessment.doc_id, step)
            # A REFUTE (inline or off-path) is a positive verification finding.
            # Mere exclusion of an unresolved MEDIUM passage is *not* counted as
            # detecting poison; it is recorded as an unreleased block instead.
            for verification in list(result.decision.verified) + list(result.queued_verifications):
                if verification.outcome.value == "REFUTE":
                    flagged_docs.add(verification.doc_id)
                    first_detection.setdefault(verification.doc_id, step)
            for doc_id in result.decision.notes.get("newly_quarantined", ()) or ():
                first_quarantine.setdefault(doc_id, step)
                flagged_docs.add(doc_id)
                first_detection.setdefault(doc_id, step)
            for doc_id in result.newly_quarantined:
                first_quarantine.setdefault(doc_id, step)
                flagged_docs.add(doc_id)
                first_detection.setdefault(doc_id, step)

            if poison_context or poison_cited:
                if result.answer_id is not None:
                    poison_answers.add(int(result.answer_id))

            ground_truth = answer_ground_truth.get(query.query_id)
            answer_success: Optional[bool] = None
            gold_correct: Optional[bool] = None
            answer_status = "NOT MEASURED"
            criterion = None
            if ground_truth:
                answer_success = attack_answer_success(
                    result.answer,
                    str(ground_truth.get("malicious_answer", "")),
                    str(ground_truth.get("gold_answer", "")),
                    gold_answers=ground_truth.get("gold_answers") or (),
                )
                gold_correct = gold_answer_present(
                    result.answer, str(ground_truth.get("gold_answer", "")),
                    ground_truth.get("gold_answers") or ())
                if answer_success is not None:
                    answer_status = "MEASURED"
                    criterion = ("attacker answer present in delivered answer and "
                                 "gold answer absent")

            records.append(QueryStepRecord(
                step=query.step, query_id=query.query_id, query=query.query,
                is_target=query.is_target, answer=result.answer,
                abstained=bool(result.record.abstained),
                abstain_reason=result.record.abstain_reason,
                retrieved_ids=retrieved, context_ids=context, excluded_ids=excluded,
                cited_ids=cited, poison_targeted=tuple(sorted(targeted)),
                poison_retrieved=poison_retrieved,
                poison_in_context=poison_context, poison_cited=poison_cited,
                bands=band_counts,
                verified=tuple((v.doc_id, _status(v.outcome)) for v in result.decision.verified),
                drained=tuple((v.doc_id, _status(v.outcome)) for v in result.queued_verifications),
                newly_quarantined=tuple(result.newly_quarantined),
                llm_calls=int(result.llm_calls),
                latency_ms=float(result.timings_ms.get("total", 0.0)),
                answer_id=result.answer_id, answer_attack_success=answer_success,
                answer_gold_correct=gold_correct,
                answer_attack_status=answer_status,
                attack_answer_criterion=criterion, result=result,
            ))

    stream_end = max((record.step for record in records), default=0)
    blocked = set()
    ledger = getattr(pipeline, "trust_provider", None)
    if ledger is not None and hasattr(ledger, "blocked_doc_ids"):
        blocked = set(ledger.blocked_doc_ids())
    for doc_id in blocked:
        flagged_docs.add(doc_id)
        first_quarantine.setdefault(doc_id, stream_end)
        first_detection.setdefault(doc_id, stream_end)

    if not declared_clean and hasattr(pipeline, "store"):
        declared_clean = {record.chunk_id for record in pipeline.store.iter_chunks()} - poison_ground_truth
    excluded_clean = {doc_id for record in records for doc_id in record.excluded_ids
                      if doc_id in declared_clean}

    metrics = _compute_metrics(
        records=records, plan=plan, stream_end=stream_end, answer_ground_truth=answer_ground_truth,
        poison_ids=poison_ground_truth, clean_ids=declared_clean,
        flagged_docs=flagged_docs, blocked=blocked,
        injection_steps=injection_steps, first_detection=first_detection,
        first_quarantine=first_quarantine, poison_answers=poison_answers,
        pipeline=pipeline,
    )
    metrics["clean_withheld_without_detection"] = len(excluded_clean)
    run = StreamRun(
        plan=plan, records=records, metrics=metrics,
        config=_run_config(pipeline),
        injection_steps=injection_steps, poison_doc_ids=poison_ground_truth,
        training_rows=training_rows,
    )
    if output_dir is not None:
        run.write(output_dir)
    return run


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _targeted_poison(query: StreamQuery, poison_by_target_query: Mapping[str, Set[str]],
                     all_poison: Set[str], injection_steps: Mapping[str, int],
                     step: int) -> Set[str]:
    if not query.is_target:
        return set()
    if query.query in poison_by_target_query:
        targeted = set(poison_by_target_query[query.query])
    elif not poison_by_target_query:
        # No per-query mapping was supplied: every poison applies to every
        # target query.  This is explicit, not a silent fallback.
        targeted = set(all_poison)
    else:
        targeted = set()
    # Only poison that has actually been ingested before this query can affect it.
    return {doc_id for doc_id in targeted if injection_steps.get(doc_id, step + 1) < step}


def _compute_metrics(*, records: Sequence[QueryStepRecord], plan: StreamPlan, stream_end: int,
                     answer_ground_truth: Mapping[str, Mapping[str, Any]],
                     poison_ids: Set[str], clean_ids: Set[str], flagged_docs: Set[str],
                     blocked: Set[str], injection_steps: Mapping[str, int],
                     first_detection: Mapping[str, int], first_quarantine: Mapping[str, int],
                     poison_answers: Set[int], pipeline: PersonAPipeline) -> Dict[str, Any]:
    target_records = [record for record in records if record.is_target]
    # Eligibility is fixed before looking at outcomes: a target query with at
    # least one targeted poison chunk ingested before the query is an attack
    # attempt, whether it abstained, failed in generation, or never retrieved
    # the poison.  Those cases stay in the denominator instead of disappearing.
    attempts = [record for record in target_records if record.poison_targeted]
    exposed_cases = [record for record in attempts if record.poison_exposed]
    cited_cases = [record for record in attempts if record.poison_cited]
    retrieved_cases = [record for record in attempts if record.poison_retrieved]
    not_exposed_after_retrieval = [record for record in retrieved_cases
                                   if not record.poison_exposed]
    never_retrieved = [record for record in attempts if not record.poison_retrieved]

    detection = detection_metrics(flagged_docs, poison_ids, clean_ids, blocked)
    # "Detected" (flagged) and "kept away from the model" are different claims.  A
    # defence can withhold poison from the context (MEDIUM band, unresolved
    # verification, trust floor) without ever escalating it, which the strict
    # detection metric deliberately does not credit.  Report both.
    retrieved_poison = {doc for record in records for doc in record.retrieved_ids
                        if doc in poison_ids}
    withheld_poison = {doc for record in records for doc in record.excluded_ids
                       if doc in poison_ids} & retrieved_poison
    retrieved_clean = {doc for record in records for doc in record.retrieved_ids
                       if doc in clean_ids}
    withheld_clean = {doc for record in records for doc in record.excluded_ids
                      if doc in clean_ids} & retrieved_clean
    delays = detection_delays(injection_steps, first_detection)
    exposures = exposure_windows(injection_steps, first_quarantine, stream_end)

    verified = [outcome for record in records for _, outcome in record.verified]
    drained = [outcome for record in records for _, outcome in record.drained]
    llm_calls = [record.llm_calls for record in records]
    latencies = [record.latency_ms for record in records if record.latency_ms > 0]
    non_target = [record for record in records if not record.is_target]
    answered_non_target = [record for record in non_target if not record.abstained]

    remediation = remediation_recall(pipeline, sorted(poison_answers))
    answer_evaluated = [record for record in attempts
                        if record.answer_attack_success is not None]
    answer_successes = sum(1 for record in answer_evaluated
                           if record.answer_attack_success is True)
    if answer_evaluated:
        answer_fields = answer_level_asr_fields(
            answer_successes / len(answer_evaluated), evaluated=len(answer_evaluated),
            status="MEASURED",
            reason=("Measured only for eligible attempts with attack-specific answer "
                    "ground truth; attacker answer present and gold answer absent. "
                    "Abstention counts as attack failure."),
        )
        answer_fields["answer_attack_criterion"] = (
            "attacker answer present in delivered answer and gold answer absent")
    else:
        answer_fields = answer_level_asr_fields()

    return {
        "queries": len(records),
        "target_queries": len(target_records),
        # Explicit metric contract.  The denominator is every eligible attack
        # attempt; see metrics.poison_exposure_rate.  These are retrieval /
        # context measurements, not claims that the answer was successfully
        # manipulated.
        "eligible_attack_attempts": len(attempts),
        "poison_exposed_attempts": len(exposed_cases),
        "poison_exposure_rate": poison_exposure_rate(len(exposed_cases), len(attempts)),
        "poison_cited_attempts": len(cited_cases),
        "poison_citation_rate": poison_citation_rate(len(cited_cases), len(attempts)),
        "poison_never_retrieved_attempts": len(never_retrieved),
        "poison_not_exposed_after_retrieval_attempts": len(not_exposed_after_retrieval),
        "poison_abstained_attempts": sum(1 for record in attempts if record.abstained),
        "target_queries_without_ingested_poison": len(target_records) - len(attempts),
        "answer_abstention_rate_on_attempts": abstention_rate(
            sum(1 for record in attempts if record.abstained), len(attempts)),
        # Measured only when the experiment supplies an attack-specific answer
        # target.  Never inferred from exposure or citation.
        **answer_fields,
        "answer_ground_truth_attempts": len(answer_evaluated),
        "answer_attack_successes": int(answer_successes),
        "answer_gold_correct_attempts": sum(1 for record in answer_evaluated
                                            if record.answer_gold_correct is True),
        "answer_gold_correct_rate": (
            sum(1 for record in answer_evaluated if record.answer_gold_correct is True)
            / len(answer_evaluated)) if answer_evaluated else None,
        "poison_leaked_cases": len(exposed_cases),
        "poison_not_exposed_after_retrieval_cases": len(not_exposed_after_retrieval),
        "poison_never_retrieved_cases": len(never_retrieved),
        "poison_leaked_docs": sorted({doc for record in exposed_cases for doc in record.poison_in_context}),
        "poison_cited_docs": sorted({doc for record in cited_cases for doc in record.poison_cited}),
        "poison_retrieved_docs": sorted({doc for record in records for doc in record.poison_retrieved}),
        "poison_quarantined": sorted(set(poison_ids) & blocked),
        "clean_quarantined": sorted(set(clean_ids) & blocked),
        "detection": detection.to_dict(),
        "withholding": {
            "retrieved_poison_docs": len(retrieved_poison),
            "withheld_poison_docs": len(withheld_poison),
            "withheld_poison_recall": (len(withheld_poison) / len(retrieved_poison)
                                       if retrieved_poison else None),
            "retrieved_clean_docs": len(retrieved_clean),
            "withheld_clean_docs": len(withheld_clean),
            "withheld_clean_rate": (len(withheld_clean) / len(retrieved_clean)
                                    if retrieved_clean else None),
        },
        "detection_delay_steps": delays,
        "mean_detection_delay_steps": summarise_optional(delays.values()),
        "exposure_window_steps": exposures,
        "max_exposure_window_steps": max(exposures.values()) if exposures else None,
        "verification": {
            "SUPPORT": verified.count("SUPPORT"),
            "REFUTE": verified.count("REFUTE"),
            "NEUTRAL": verified.count("NEUTRAL"),
            "queued_SUPPORT": drained.count("SUPPORT"),
            "queued_REFUTE": drained.count("REFUTE"),
            "queued_NEUTRAL": drained.count("NEUTRAL"),
            "errors": sum(
                1 for record in records for result in
                (_verification_results(record))
                if getattr(result, "error", None)
            ),
        },
        "remediation": {
            "poison_answers": len(poison_answers),
            "flagged_poison_answers": sum(
                1 for answer_id in poison_answers
                if (stored := pipeline.answer_log.get(answer_id)) is not None and stored.flagged
            ),
            "remediation_recall": remediation,
        },
        "abstention_rate": abstention_rate(sum(1 for record in records if record.abstained),
                                            len(records)),
        "clean_utility": clean_utility(len(answered_non_target), len(non_target)),
        "llm_calls": {
            "mean_per_query": mean_llm_calls(llm_calls),
            "total": int(sum(llm_calls)),
        },
        "latency_ms": latency_stats(latencies),
        "policy": type(getattr(pipeline, "policy", None)).__name__,
        "verifier": type(getattr(pipeline, "verifier", None)).__name__
        if getattr(pipeline, "verifier", None) is not None else None,
        "verifier_mode": getattr(getattr(pipeline, "verifier", None), "mode", None),
        "ledger_docs": len(blocked),
        "stream_end": int(stream_end),
        "n_ingestions": len(plan.ingestions),
        "n_poison_ingestions": sum(1 for event in plan.ingestions if event.is_poison),
        "n_warmup_ingestions": sum(1 for event in plan.ingestions if not event.is_poison),
    }


def _verification_results(record: QueryStepRecord) -> List[VerificationResult]:
    result = record.result
    if result is None:
        return []
    return list(result.decision.verified) + list(result.queued_verifications)


def remediation_recall(pipeline: PersonAPipeline, answer_ids: Sequence[int]) -> Optional[float]:
    """Fraction of exposed answers that retroactive remediation flagged."""
    if not answer_ids:
        return None
    flagged = 0
    for answer_id in answer_ids:
        stored = pipeline.answer_log.get(int(answer_id))
        if stored is not None and stored.flagged:
            flagged += 1
    return float(flagged) / float(len(answer_ids))


def _run_config(pipeline: PersonAPipeline) -> Dict[str, Any]:
    config = pipeline.config
    health = pipeline.health_check()
    return {
        "seed": config.seed,
        "time": config.time.model_dump(mode="json"),
        "embedding": {"backend": config.embedding.backend,
                      "model_name": config.embedding.model_name,
                      "dim": getattr(pipeline.embedder, "dim", None)},
        "index": config.index.model_dump(mode="json"),
        "retrieval": config.retrieval.model_dump(mode="json"),
        "generation": {"backend": config.generation.backend,
                       "model_name": config.generation.model_name,
                       "abstain_evidence_mass": config.generation.abstain_evidence_mass},
        "security": config.security.model_dump(mode="json"),
        "trust": config.trust.model_dump(mode="json"),
        "health": health,
        "scorer": getattr(pipeline.scorer, "version", "unknown"),
        "policy": type(getattr(pipeline, "policy", None)).__name__,
        "verifier_mode": getattr(getattr(pipeline, "verifier", None), "mode", None),
        "llm": getattr(getattr(pipeline.generator, "llm", None), "name", "unknown"),
    }


def _summary_markdown(run: StreamRun) -> str:
    m = run.metrics
    lines = [
        "# Chronological stream summary",
        "",
        f"- queries: {m['queries']} ({m['target_queries']} target)",
        f"- eligible attack attempts: {m['eligible_attack_attempts']}",
        f"- poison exposure rate: {m['poison_exposure_rate']} "
        f"({m['poison_exposed_attempts']}/{m['eligible_attack_attempts']})",
        f"- poison citation rate: {m['poison_citation_rate']} "
        f"({m['poison_cited_attempts']}/{m['eligible_attack_attempts']})",
        f"- answer-level ASR: {m['answer_level_asr']} "
        f"({m['answer_level_asr_status']})",
        f"- poison never retrieved: {m['poison_never_retrieved_attempts']}",
        f"- poison not exposed after retrieval: "
        f"{m['poison_not_exposed_after_retrieval_attempts']}",
        f"- detection precision/recall/F1: {m['detection']['precision']} / "
        f"{m['detection']['recall']} / {m['detection']['f1']}",
        f"- clean false positives: {m['detection']['false_positives']} "
        f"(rate {m['detection']['false_positive_rate']})",
        f"- quarantined poison / clean: {len(m['poison_quarantined'])} / "
        f"{len(m['clean_quarantined'])}",
        f"- remediation recall: {m['remediation']['remediation_recall']}",
        f"- abstention rate: {m['abstention_rate']}",
        f"- clean utility: {m['clean_utility']}",
        f"- mean LLM calls/query: {m['llm_calls']['mean_per_query']}",
        f"- latency p50/p95 ms: {m['latency_ms']['p50_ms']} / {m['latency_ms']['p95_ms']}",
        "",
        f"Policy: `{m['policy']}`; verifier: `{m['verifier']}` "
        f"(mode `{m['verifier_mode']}`); health: `{run.config.get('health', {}).get('status')}`.",
        "",
        "This summary is generated from observed events.  It is a smoke-test result "
        "unless real embedding, generation and NLI backends were configured.",
    ]
    return "\n".join(lines) + "\n"
