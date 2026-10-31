"""Adapters between Person C streams and the A/B pipeline.

All conversions go through :mod:`trace_rag.attacks.stream`, which owns the
canonical row shapes and validates required fields.  The previous version of
this module emitted ``{text, qid, t}`` while ``chronological.run_stream`` read
``{query, query_id, step}``, so the documented entry point crashed with
``KeyError: 'step'``.  There is now one contract, and consumers call
:func:`as_query_row` / :func:`as_ingestion_row` rather than indexing raw dicts.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from trace_rag.attacks.stream import (
    StreamIngestion,
    StreamQuery,
    as_ingestion_row,
    as_query_row,
)


def to_pipeline_queries(queries: Iterable[Any]) -> List[Mapping[str, Any]]:
    """Convert stream queries (objects or rows) to canonical pipeline rows.

    Output shape: ``{"step": int, "query_id": str, "query": str}``.
    """
    return [as_query_row(query, index=i) for i, query in enumerate(queries)]


def to_pipeline_events(events: Iterable[Any]) -> List[Mapping[str, Any]]:
    """Convert ingestion events (objects or rows) to canonical event rows."""
    return [as_ingestion_row(event) for event in events]


def to_stream_queries(rows: Iterable[Any]) -> List[StreamQuery]:
    """Canonical rows -> :class:`StreamQuery` objects (for plan validation)."""
    out: List[StreamQuery] = []
    for i, row in enumerate(rows):
        canonical = as_query_row(row, index=i)
        original = row if isinstance(row, Mapping) else {}
        target = original.get("target_query") if isinstance(original, Mapping) else getattr(row, "target_query", None)
        is_target = bool(original.get("is_target", False)) if isinstance(original, Mapping) \
            else bool(getattr(row, "is_target", False))
        out.append(StreamQuery(step=canonical["step"], query_id=canonical["query_id"],
                               query=canonical["query"], target_query=target,
                               is_target=is_target))
    return out


def to_stream_ingestions(rows: Iterable[Any]) -> List[StreamIngestion]:
    """Canonical rows -> :class:`StreamIngestion` objects."""
    out: List[StreamIngestion] = []
    for row in rows:
        canonical = as_ingestion_row(row)
        out.append(StreamIngestion(**canonical))
    return out


__all__ = ["to_pipeline_queries", "to_pipeline_events", "to_stream_queries",
           "to_stream_ingestions"]
