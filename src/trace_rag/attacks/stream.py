"""Chronological query-stream simulation for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StreamQuery:
    """One query arriving at a specific point in the evaluation stream."""

    step: int
    query_id: str
    query: str
    target_query: Optional[str] = None
    is_target: bool = False


@dataclass(frozen=True)
class StreamIngestion:
    """One document ingestion event in chronological order."""

    step: int
    doc_id: str
    text: str
    source_id: str
    family_id: Optional[str] = None
    is_poison: bool = False


@dataclass
class StreamResult:
    """Recorded outcome of running a chronological query stream."""

    queries: List[StreamQuery]
    ingestions: List[StreamIngestion]
    answers: List[object]

def zipf_queries(
    queries: List[str],
    n: int,
    seed: int = 20260921,
) -> List[str]:
    """Generate a deterministic query stream with Zipf-like repetition."""
    if not queries:
        raise ValueError("queries must not be empty")
    if n <= 0:
        raise ValueError("n must be positive")

    import numpy as np

    rng = np.random.default_rng(seed)
    ranks = np.arange(1, len(queries) + 1, dtype=np.float64)
    weights = 1.0 / ranks
    probabilities = weights / weights.sum()

    indices = rng.choice(len(queries), size=n, p=probabilities)
    return [queries[int(i)] for i in indices]

def gradual_schedule(
    total_steps: int,
    poison_start: int,
    poison_every: int = 10,
) -> List[int]:
    """Return chronological steps at which poison is injected."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if poison_start < 0 or poison_start >= total_steps:
        raise ValueError("poison_start must fall inside the stream")
    if poison_every <= 0:
        raise ValueError("poison_every must be positive")

    return list(range(poison_start, total_steps, poison_every))

def build_query_stream(
    queries: List[str],
    total_steps: int,
    poison_start: int,
    poison_every: int = 10,
    target_query: Optional[str] = None,
    seed: int = 20260921,
) -> List[StreamQuery]:
    """Build a chronological query stream with repeated target queries."""
    sampled = zipf_queries(queries, total_steps, seed=seed)
    poison_steps = set(
        gradual_schedule(total_steps, poison_start, poison_every)
    )

    stream: List[StreamQuery] = []

    for step, query in enumerate(sampled):
        is_target = (
            target_query is not None
            and query.strip().lower() == target_query.strip().lower()
        )

        stream.append(
            StreamQuery(
                step=step,
                query_id=f"stream_q_{step:06d}",
                query=query,
                target_query=target_query,
                is_target=is_target and step in poison_steps,
            )
        )

    return stream

def build_ingestion_schedule(
    clean_documents: List[tuple[str, str]],
    poison_documents: List[tuple[str, str]],
    poison_start: int,
    poison_every: int = 10,
    clean_source_prefix: str = "clean_src",
    poison_source_prefix: str = "attacker",
) -> List[StreamIngestion]:
    """Schedule clean and poison documents over the same chronological stream."""
    if not clean_documents:
        raise ValueError("clean_documents must not be empty")
    if poison_documents and poison_start < 0:
        raise ValueError("poison_start must be non-negative")
    if poison_every <= 0:
        raise ValueError("poison_every must be positive")

    events: List[StreamIngestion] = []

    # Benign contributors continue contributing throughout the stream.
    for i, (doc_id, text) in enumerate(clean_documents):
        events.append(
            StreamIngestion(
                step=i,
                doc_id=doc_id,
                text=text,
                source_id=f"{clean_source_prefix}_{i % 3}",
                is_poison=False,
            )
        )

    # Poison arrives gradually instead of being inserted all at once.
    for i, (doc_id, text) in enumerate(poison_documents):
        events.append(
            StreamIngestion(
                step=poison_start + i * poison_every,
                doc_id=doc_id,
                text=text,
                source_id=poison_source_prefix,
                is_poison=True,
            )
        )

    return sorted(events, key=lambda event: (event.step, event.doc_id))
