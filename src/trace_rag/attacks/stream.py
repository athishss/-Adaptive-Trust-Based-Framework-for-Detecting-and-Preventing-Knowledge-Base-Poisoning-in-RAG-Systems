"""Chronological query and ingestion schedules for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class StreamQuery:
    """One query scheduled at a discrete stream step."""

    step: int
    query_id: str
    query: str
    target_query: Optional[str] = None
    is_target: bool = False

    def to_dict(self) -> dict:
        return {
            "step": int(self.step),
            "query_id": self.query_id,
            "query": self.query,
            "target_query": self.target_query,
            "is_target": bool(self.is_target),
        }


@dataclass(frozen=True)
class StreamIngestion:
    """One passage scheduled for ingestion before a stream query."""

    step: int
    doc_id: str
    text: str
    source_id: str
    family_id: Optional[str] = None
    is_poison: bool = False
    attack_type: str = "unknown"
    target_query: Optional[str] = None
    title: str = ""
    origin: str = "person_c"

    def to_dict(self) -> dict:
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
        }


@dataclass(frozen=True)
class StreamResult:
    """Serializable record of a chronological evaluation run."""

    queries: Sequence[StreamQuery]
    ingestions: Sequence[StreamIngestion]
    answers: Sequence[object]

    def to_dict(self) -> dict:
        return {
            "queries": [q.to_dict() for q in self.queries],
            "ingestions": [i.to_dict() for i in self.ingestions],
            "answers": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.answers],
        }


def zipf_queries(queries: Sequence[str], n: int, seed: int = 0,
                 exponent: float = 1.0) -> List[str]:
    """Sample a reproducible query stream with a Zipf popularity curve.

    Rank-one queries are ``1 / rank**exponent`` as likely as the least popular
    ones. The returned list preserves the original strings and has exactly *n*
    entries, allowing attack scripts to replay repeated targets.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    population = [str(query) for query in queries if str(query).strip()]
    if n == 0:
        return []
    if not population:
        raise ValueError("queries must contain at least one non-empty query")
    weights = [1.0 / (rank ** exponent) for rank in range(1, len(population) + 1)]
    return random.Random(seed).choices(population, weights=weights, k=n)


def build_query_stream(queries: Sequence[str], total_steps: int, seed: int = 0,
                       exponent: float = 1.0,
                       target_queries: Sequence[str] = (),
                       target_start: int = 0, target_every: int = 1
                       ) -> List[StreamQuery]:
    """Build a replayable Zipf stream with designated target-query steps.

    Target queries cycle through ``target_queries`` on steps
    ``target_start, target_start + target_every, ...``. Other steps are sampled
    from the remaining query population, so targets are not accidentally
    marked as ordinary traffic. Ingestion order is checked separately by
    :func:`validate_target_order` when the schedules are joined.
    """
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    if target_start < 0:
        raise ValueError("target_start must be non-negative")
    if target_every <= 0:
        raise ValueError("target_every must be positive")
    population = [str(query) for query in queries if str(query).strip()]
    targets = [str(query) for query in target_queries if str(query).strip()]
    if total_steps == 0:
        return []
    if not population and not targets:
        raise ValueError("queries must contain at least one non-empty query")
    scheduled_targets = set(range(target_start, total_steps, target_every)) if targets else set()
    ordinary = [query for query in population if query not in set(targets)] or population or targets
    sampled = iter(zipf_queries(ordinary, total_steps, seed=seed, exponent=exponent))
    result: List[StreamQuery] = []
    target_index = 0
    for step in range(total_steps):
        if step in scheduled_targets:
            target = targets[target_index % len(targets)]
            target_index += 1
            result.append(StreamQuery(step, f"q_{step}", target,
                                      target_query=target, is_target=True))
        else:
            result.append(StreamQuery(step, f"q_{step}", next(sampled)))
    return result


def build_ingestion_schedule(payloads: Sequence[str], *, start_step: int = 0,
                             interval: int = 1, source_id: str = "attacker",
                             family_id: Optional[str] = None,
                             doc_prefix: str = "injected",
                             is_poison: bool = True,
                             attack_type: str = "injection",
                             target_queries: Sequence[Optional[str]] = (),
                             title: str = "", origin: str = "person_c"
                             ) -> List[StreamIngestion]:
    """Place passages on a deterministic ingestion schedule.

    ``target_queries`` associates each injected passage with the question it is
    intended to affect. Schedulers should choose ``start_step`` no later than
    that query's step; :func:`validate_target_order` rejects late injections.
    """
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if not source_id or not doc_prefix:
        raise ValueError("source_id and doc_prefix must not be empty")
    if any(not str(payload).strip() for payload in payloads):
        raise ValueError("payloads must not contain empty text")
    return [
        StreamIngestion(
            step=start_step + i * interval,
            doc_id=f"{doc_prefix}_{i}",
            text=str(payload),
            source_id=source_id,
            family_id=family_id,
            is_poison=is_poison,
            attack_type=attack_type,
            target_query=(target_queries[i] if i < len(target_queries) else None),
            title=title,
            origin=origin,
        )
        for i, payload in enumerate(payloads)
    ]


def validate_target_order(events: Sequence[dict], queries: Sequence[dict]) -> None:
    """Reject target queries that precede the poison event intended for them."""
    event_steps: dict[str, int] = {}
    for event in events:
        target = event.get("target_query")
        if target and event.get("is_poison", False):
            event_steps[str(target)] = min(event_steps.get(str(target), int(event["step"])),
                                           int(event["step"]))
    for query in queries:
        if not query.get("is_target", False):
            continue
        target = query.get("target_query") or query.get("query")
        if target is None or str(target) not in event_steps:
            continue
        if event_steps[str(target)] > int(query["step"]):
            raise ValueError(
                f"poison for target query {target!r} is ingested at step "
                f"{event_steps[str(target)]}, after its query at step {query['step']}"
            )
