"""Chronological Person C evaluation harness using Person A ingestion/pipeline APIs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_rag.ingestion.pipeline import Ingestor
from trace_rag.pipeline import PersonAPipeline


def run_stream(
    pipeline: PersonAPipeline,
    ingestor: Ingestor,
    events: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> list[object]:
    """Ingest events chronologically and answer queries at their stream step.

    Events at a step are ingested and indexed before queries at that same step
    are evaluated. This guarantees that a poison event cannot affect a query
    before the poison has entered the corpus.
    """
    answers: list[object] = []

    events_by_step: dict[int, list[Mapping[str, Any]]] = {}
    queries_by_step: dict[int, list[Mapping[str, Any]]] = {}

    for event in events:
        events_by_step.setdefault(int(event["step"]), []).append(event)

    for query in queries:
        queries_by_step.setdefault(int(query["step"]), []).append(query)

    all_steps = sorted(set(events_by_step) | set(queries_by_step))

    for step in all_steps:
        new_chunk_ids: list[str] = []

        # 1. Ingest all documents scheduled for this step.
        for event in events_by_step.get(step, []):
            records = ingestor.ingest_text(
                doc_id=str(event["doc_id"]),
                text=str(event["text"]),
                source_id=str(event["source_id"]),
                title=str(event.get("title", "")),
                origin=str(event.get("origin", "person_c")),
                ingested_at=float(step),
                metadata={
                    "is_poison": bool(event.get("is_poison", False)),
                    "family_id": event.get("family_id"),
                    "attack_type": event.get("attack_type"),
                },
            )

            new_chunk_ids.extend(record.chunk_id for record in records)

        # 2. Index newly ingested chunks before answering queries.
        if new_chunk_ids:
            pipeline.index_chunks(chunk_ids=new_chunk_ids)

        # 3. Only now answer queries occurring at this step.
        for query in queries_by_step.get(step, []):
            result = pipeline.answer(
                query=str(query["query"]),
                query_id=str(query.get("query_id", f"q_{step}")),
                step=step,
            )
            answers.append(result)

    return answers
