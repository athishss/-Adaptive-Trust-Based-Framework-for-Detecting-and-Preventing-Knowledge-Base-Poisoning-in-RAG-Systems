"""Small chronological Person C evaluation harness."""

from __future__ import annotations

import hashlib
from pathlib import Path

from trace_rag.ingestion.provenance import ProvenanceStore
from trace_rag.pipeline import PersonAPipeline


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_stream(
    pipeline: PersonAPipeline,
    store: ProvenanceStore,
    events: list[dict],
    queries: list[dict],
) -> list[object]:
    """Ingest/index documents and answer queries in chronological order."""

    answers = []

    events_by_step: dict[int, list[dict]] = {}
    for event in events:
        events_by_step.setdefault(int(event["step"]), []).append(event)

    queries_by_step: dict[int, list[dict]] = {}
    for query in queries:
        queries_by_step.setdefault(int(query["step"]), []).append(query)

    all_steps = sorted(set(events_by_step) | set(queries_by_step))

    for step in all_steps:
        new_chunk_ids = []

        for event in events_by_step.get(step, []):
            doc_id = str(event["doc_id"])
            source_id = str(event["source_id"])
            text = str(event["text"])

            pipeline.ingest_document(
                doc_id=doc_id,
                source_id=source_id,
                text=text,
                title=str(event.get("title", "")),
                origin=str(event.get("origin", "person_c")),
                metadata={
                    "is_poison": bool(event.get("is_poison", False)),
                    "family_id": event.get("family_id"),
                },
                ingested_at=float(step),
            )

            chunks = store.get_chunks_for_document(doc_id)
            new_chunk_ids.extend(chunk.chunk_id for chunk in chunks)

        if new_chunk_ids:
            pipeline.index_chunks(chunk_ids=new_chunk_ids)

        for query in queries_by_step.get(step, []):
            result = pipeline.answer(
                query=str(query["query"]),
                query_id=str(query.get("query_id", f"q_{step}")),
                step=step,
            )
            answers.append(result)

    return answers
