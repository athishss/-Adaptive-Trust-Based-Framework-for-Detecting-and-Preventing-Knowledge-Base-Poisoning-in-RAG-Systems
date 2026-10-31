"""Chronological query-stream simulation and the canonical stream contracts.

This module owns the *shape* of stream data.  Three consumers used to disagree
about it - ``evaluation.integration.to_pipeline_queries`` emitted
``{text, qid, t}`` while ``evaluation.chronological.run_stream`` read
``{query, query_id, step}``, and the ingestion side had no adapter at all.

The canonical row shapes are now defined once here:

* query row:   ``{"step": int, "query_id": str, "query": str}``
* ingestion:   ``{"step": int, "doc_id": str, "text": str, "source_id": str, ...}``

:func:`as_query_row` and :func:`as_ingestion_row` accept the dataclasses below,
the legacy aliases (``t``/``qid``/``text``) and validate required fields.  Every
runner and adapter calls them, so a schema change cannot silently desynchronise
the harness again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

QUERY_ROW_KEYS = ("step", "query_id", "query")
INGESTION_ROW_KEYS = ("step", "doc_id", "text", "source_id")


class StreamContractError(ValueError):
    """A stream row does not satisfy the canonical contract."""


@dataclass(frozen=True)
class StreamQuery:
    step: int
    query_id: str
    query: str
    target_query: Optional[str] = None
    is_target: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "query_id": self.query_id,
            "query": self.query,
            "target_query": self.target_query,
            "is_target": bool(self.is_target),
        }

    def as_row(self) -> Dict[str, Any]:
        return {"step": int(self.step), "query_id": self.query_id, "query": self.query}


@dataclass(frozen=True)
class StreamIngestion:
    step: int
    doc_id: str
    text: str
    source_id: str
    family_id: Optional[str] = None
    is_poison: bool = False
    attack_type: Optional[str] = None
    target_query: Optional[str] = None
    title: str = ""
    origin: str = "person_c"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "doc_id": self.doc_id,
            "text": self.text,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "is_poison": bool(self.is_poison),
            "attack_type": self.attack_type,
            "target_query": self.target_query,
            "title": self.title,
            "origin": self.origin,
            "metadata": dict(self.metadata),
        }

    def as_row(self) -> Dict[str, Any]:
        return self.to_dict()


@dataclass
class StreamResult:
    queries: List[StreamQuery]
    ingestions: List[StreamIngestion]
    answers: List[object]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries": [q.to_dict() for q in self.queries],
            "ingestions": [i.to_dict() for i in self.ingestions],
            "answers": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.answers],
        }


# ---------------------------------------------------------------------------
#  Canonical adapters
# ---------------------------------------------------------------------------


def _get(row: Any, *names: str, default: Any = None, required: bool = False,
         label: str = "") -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
    else:
        for name in names:
            if hasattr(row, name):
                value = getattr(row, name)
                if value is not None:
                    return value
    if required:
        raise StreamContractError(
            f"missing required field {label or names[0]!r} (accepted names: {names})"
        )
    return default


def as_query_row(row: Any, index: int = 0) -> Dict[str, Any]:
    """Normalise any accepted query representation to the canonical row."""
    step = _get(row, "step", "t", "timestep", default=index, required=True, label="step")
    query_id = _get(row, "query_id", "qid", "id", default=f"q{int(step)}", required=False)
    query = _get(row, "query", "text", "question", required=True, label="query")
    try:
        step_int = int(step)
    except (TypeError, ValueError) as exc:
        raise StreamContractError(f"step must be an integer, got {step!r}") from exc
    if step_int < 0:
        raise StreamContractError(f"step must be >= 0, got {step_int}")
    if not str(query).strip():
        raise StreamContractError("query must be a non-empty string")
    return {"step": step_int, "query_id": str(query_id), "query": str(query)}


def as_ingestion_row(row: Any) -> Dict[str, Any]:
    """Normalise any accepted ingestion representation to the canonical row."""
    data = {
        "step": int(_get(row, "step", "t", "timestep", required=True, label="step")),
        "doc_id": str(_get(row, "doc_id", "id", "document_id", required=True)),
        "text": str(_get(row, "text", "payload", "content", required=True)),
        "source_id": str(_get(row, "source_id", "source", required=True)),
        "family_id": _get(row, "family_id", "family"),
        "is_poison": bool(_get(row, "is_poison", "poison", default=False)),
        "attack_type": _get(row, "attack_type", "attack"),
        "target_query": _get(row, "target_query"),
        "title": str(_get(row, "title", default="") or ""),
        "origin": str(_get(row, "origin", default="person_c") or "person_c"),
        "metadata": dict(_get(row, "metadata", default={}) or {}),
    }
    if data["step"] < 0:
        raise StreamContractError(f"step must be >= 0, got {data['step']}")
    if not data["doc_id"] or not data["source_id"]:
        raise StreamContractError("doc_id and source_id must be non-empty")
    if not data["text"].strip():
        raise StreamContractError(f"text must be non-empty (doc_id={data['doc_id']})")
    return data


# ---------------------------------------------------------------------------
#  Plan + ordering validation
# ---------------------------------------------------------------------------


@dataclass
class StreamPlan:
    """Queries and ingestions with a validated ordering guarantee."""

    queries: List[StreamQuery]
    ingestions: List[StreamIngestion]

    def validate(self) -> "StreamPlan":
        """Ensure poison arrives before the first query it targets.

        Detection delay and exposure window are measured in stream steps, so an
        attack that is ingested *after* its target query would make both
        undefined.  This is a hard error, not a warning.
        """
        query_ids = [q.query_id for q in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise StreamContractError("duplicate query_id in stream")
        doc_ids = [i.doc_id for i in self.ingestions]
        if len(doc_ids) != len(set(doc_ids)):
            raise StreamContractError("duplicate doc_id in ingestion schedule")

        target_queries = [q for q in self.queries if q.is_target]
        if self.ingestions and not self.queries:
            raise StreamContractError("a plan with ingestions must have queries")
        for event in self.ingestions:
            if not event.is_poison:
                continue
            candidates = [q for q in target_queries
                          if event.target_query is None or q.query == event.target_query]
            if not candidates:
                if event.target_query:
                    raise StreamContractError(
                        f"poison {event.doc_id} targets a query that is not in the "
                        f"stream: {event.target_query!r}"
                    )
                candidates = target_queries
            if candidates and event.step >= min(q.step for q in candidates):
                first = min(q.step for q in candidates)
                raise StreamContractError(
                    f"poison {event.doc_id} is ingested at step {event.step}, not "
                    f"before its first target query at step {first}"
                )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries": [q.to_dict() for q in self.queries],
            "ingestions": [i.to_dict() for i in self.ingestions],
        }


# ---------------------------------------------------------------------------
#  Query sampling
# ---------------------------------------------------------------------------


def zipf_queries(questions: Sequence[str], steps: int, seed: int = 20260921,
                 exponent: float = 1.1) -> List[str]:
    """Sample *steps* questions with a Zipf-like repetition pattern.

    Deterministic for a fixed seed.  Heavy repetition of a few questions is
    what lets trust evolve within a short stream; uniform sampling would make
    every source look equally familiar.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    questions = [str(q) for q in questions]
    if not questions:
        raise ValueError("questions must not be empty")
    if steps == 0:
        return []
    ranks = np.arange(1, len(questions) + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, float(exponent))
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(questions), size=int(steps), p=probabilities)
    return [questions[int(i)] for i in indices]


def build_query_stream(
    questions: Sequence[str],
    total_steps: int,
    *,
    target_questions: Optional[Sequence[str]] = None,
    target_start: int = 5,
    target_every: int = 4,
    seed: int = 20260921,
    query_id_prefix: str = "q",
) -> List[StreamQuery]:
    """Build a chronological query stream with explicit target queries.

    Target questions are placed on a fixed grid starting at ``target_start``;
    every other step is filled with a Zipf-sampled benign question.  The
    ingestion schedule must be built with :func:`build_stream_plan` so poison is
    placed before the first target step.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    questions = [str(q) for q in questions]
    if not questions:
        raise ValueError("questions must not be empty")
    targets = list(dict.fromkeys(str(q) for q in (target_questions or questions)))
    target_steps = list(range(int(target_start), int(total_steps), max(1, int(target_every))))
    target_at = {step: targets[i % len(targets)] for i, step in enumerate(target_steps)}
    background = zipf_queries(questions, total_steps, seed=seed)

    out: List[StreamQuery] = []
    for step in range(int(total_steps)):
        question = target_at.get(step, background[step])
        is_target = step in target_at
        out.append(StreamQuery(
            step=step, query_id=f"{query_id_prefix}{step:04d}", query=question,
            target_query=question if is_target else None, is_target=is_target,
        ))
    return out


# ---------------------------------------------------------------------------
#  Ingestion schedules
# ---------------------------------------------------------------------------


def build_ingestion_schedule(
    documents: Iterable[Any],
    *,
    first_step: int = 0,
    step_interval: int = 1,
    poison_start: Optional[int] = None,
    poison_interval: int = 0,
) -> List[StreamIngestion]:
    """Place clean and malicious ingestions on the stream timeline.

    Clean documents start at ``first_step``; poison documents start at
    ``poison_start`` (default: one step before the first clean step + count).
    """
    if step_interval < 1:
        raise ValueError("step_interval must be >= 1")
    rows = [as_ingestion_row(doc) for doc in documents]
    clean = [row for row in rows if not row["is_poison"]]
    poison = [row for row in rows if row["is_poison"]]

    events: List[StreamIngestion] = []
    for i, row in enumerate(clean):
        events.append(StreamIngestion(step=first_step + i * step_interval, **{
            key: row[key] for key in ("doc_id", "text", "source_id", "family_id",
                                     "is_poison", "attack_type", "target_query",
                                     "title", "origin", "metadata")
        }))
    base = poison_start if poison_start is not None else first_step + max(1, len(clean)) * step_interval
    spacing = poison_interval if poison_interval > 0 else step_interval
    for i, row in enumerate(poison):
        events.append(StreamIngestion(step=int(base) + i * spacing, **{
            key: row[key] for key in ("doc_id", "text", "source_id", "family_id",
                                     "is_poison", "attack_type", "target_query",
                                     "title", "origin", "metadata")
        }))
    return events


def build_stream_plan(
    queries: Sequence[StreamQuery],
    ingestions: Sequence[StreamIngestion],
) -> StreamPlan:
    """Combine and validate a query stream and an ingestion schedule."""
    return StreamPlan(queries=list(queries), ingestions=list(ingestions)).validate()
