"""Adapters between Person C stream schedules and the A/B pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_rag.attacks.stream import StreamIngestion, StreamQuery


def to_pipeline_queries(queries: Sequence[StreamQuery]) -> list[Mapping[str, Any]]:
    """Convert typed stream queries to the exact rows consumed by run_stream."""
    return [
        {
            "step": int(item.step),
            "query": item.query,
            "query_id": item.query_id,
            "target_query": item.target_query,
            "is_target": bool(item.is_target),
        }
        for item in queries
    ]


def to_pipeline_events(ingestions: Sequence[StreamIngestion]) -> list[Mapping[str, Any]]:
    """Convert typed ingestion events to the exact rows consumed by run_stream."""
    return [
        {
            "step": int(item.step),
            "doc_id": item.doc_id,
            "text": item.text,
            "source_id": item.source_id,
            "family_id": item.family_id,
            "is_poison": bool(item.is_poison),
            "attack_type": item.attack_type,
            "target_query": item.target_query,
            "title": item.title,
            "origin": item.origin,
        }
        for item in ingestions
    ]
