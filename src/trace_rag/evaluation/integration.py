"""Adapters between Person C streams and the A/B pipeline."""

from __future__ import annotations

from typing import List, Mapping, Any

from trace_rag.attacks.stream import StreamQuery


def to_pipeline_queries(
    queries: List[StreamQuery],
) -> List[Mapping[str, Any]]:
    """Convert chronological StreamQuery objects to pipeline rows."""
    return [
        {
            "text": item.query,
            "qid": item.query_id,
            "t": item.step,
        }
        for item in queries
    ]
