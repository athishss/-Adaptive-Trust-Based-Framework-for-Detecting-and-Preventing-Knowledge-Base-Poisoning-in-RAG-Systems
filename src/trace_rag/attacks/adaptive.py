"""Adaptive trust-aware attack schedules for Person C.

The adaptive attacks are about *timing and provenance*, not just text:

===============  =================================================================
attack           actual behaviour
===============  =================================================================
hit_and_run      every malicious payload arrives from a *fresh* source, so no
                 single contributor accumulates a history that could be audited
slow_burn        benign documents from the attacker's source are ingested first,
                 then malicious documents from the *same* source later, so any
                 reputation the ledger grants is earned before the attack
framing          generates rival passages that contradict a legitimate target
                 passage from several distinct fresh sources, so the legitimate
                 document looks uncorroborated / isolated
signal_evasion   rewrites the payload to remove the target query's literal echo
                 (signal S1); it deliberately does not claim to evade S3/S6,
                 which need source-level coordination rather than text edits
===============  =================================================================

Every event carries ``is_poison`` and ``phase`` so the evaluation harness can
measure the *attack* separately from its *warm-up*.  Warm-up documents are not
poison: counting them as successful attacks is one of the ways an evaluation
can silently become invalid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_WARMUP: Tuple[str, ...] = (
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
    "The Eiffel Tower is located in Paris, France.",
    "Photosynthesis takes place in the chloroplasts of plant cells.",
    "Mount Everest is the highest mountain above sea level.",
    "The Amazon rainforest spans nine countries in South America.",
)


@dataclass(frozen=True)
class AttackEvent:
    """One ingestion event in chronological order.

    ``step`` is the stream step at which the document is ingested.  A malicious
    event must be ingested strictly before the step of the query it targets;
    :mod:`trace_rag.evaluation.chronological` validates that ordering.
    """

    step: int
    attack_type: str
    source_id: str
    family_id: str
    payload: str
    doc_id: str = ""
    is_poison: bool = True
    phase: str = "attack"                 # 'warmup' | 'attack'
    target_query: Optional[str] = None
    target_doc_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be >= 0")
        if not self.payload:
            raise ValueError("payload must not be empty")
        if self.is_poison and self.phase != "attack":
            raise ValueError("a poison event must have phase='attack'")
        if not self.is_poison and self.phase != "warmup":
            raise ValueError("a non-poison event must have phase='warmup'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "attack_type": self.attack_type,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "payload": self.payload,
            "doc_id": self.doc_id or f"{self.attack_type}_{self.source_id}_{self.step}",
            "is_poison": bool(self.is_poison),
            "phase": self.phase,
            "target_query": self.target_query,
            "target_doc_id": self.target_doc_id,
            "metadata": dict(self.metadata),
        }


def _event(step: int, attack_type: str, source_id: str, family_id: str,
           payload: str, *, doc_id: str = "", is_poison: bool = True,
           phase: str = "attack", target_query: Optional[str] = None,
           target_doc_id: Optional[str] = None,
           metadata: Optional[Mapping[str, Any]] = None) -> AttackEvent:
    return AttackEvent(
        step=int(step), attack_type=attack_type, source_id=source_id,
        family_id=family_id, payload=payload,
        doc_id=doc_id or f"{attack_type}_{source_id}_{step}",
        is_poison=is_poison, phase=phase, target_query=target_query,
        target_doc_id=target_doc_id, metadata=dict(metadata or {}),
    )


def hit_and_run(
    payloads: List[str],
    start_step: int = 0,
    source_prefix: str = "attacker",
    *,
    spacing: int = 1,
    target_queries: Optional[Sequence[str]] = None,
) -> List[AttackEvent]:
    """Inject each payload from a fresh source to avoid reputation buildup.

    One document per source is the defining property: the attacker never
    accumulates a contribution history, so source-level defences must rely on
    burst/coordination detection rather than source age.
    """
    if not payloads:
        return []
    if spacing < 1:
        raise ValueError("spacing must be >= 1")
    events: List[AttackEvent] = []
    for i, payload in enumerate(payloads):
        source_id = f"{source_prefix}_{i}"
        events.append(_event(
            start_step + i * spacing, "hit_and_run", source_id,
            f"{source_prefix}_family_{i}", payload,
            doc_id=f"{source_id}_doc", target_query=(target_queries[i] if target_queries else None),
            metadata={"lifetime_steps": 1, "distribution": "fresh_source_per_document"},
        ))
    return events


def slow_burn(
    payloads: List[str],
    warmup_steps: int = 5,
    source_id: str = "attacker",
    family_id: str = "attacker_family",
    *,
    start_step: int = 0,
    warmup_payloads: Optional[Sequence[str]] = None,
    target_queries: Optional[Sequence[str]] = None,
) -> List[AttackEvent]:
    """Build a source's benign history before injecting malicious payloads.

    ``warmup_steps`` warm-up documents from ``source_id`` are ingested at steps
    ``start_step .. start_step + warmup_steps - 1``; the malicious payloads
    follow at ``start_step + warmup_steps + i``.  The warm-up documents are
    labelled ``is_poison=False``: they are the reputation the attacker is
    trying to earn, not attack successes.

    Whether that reputation actually accrues depends on the defence observing
    the benign documents.  The harness must run benign queries that retrieve
    them; ingesting text alone does not create verified trust.
    """
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    if not payloads:
        return []
    warmup_texts = list(warmup_payloads or DEFAULT_WARMUP)
    if not warmup_texts:
        raise ValueError("warmup_payloads must not be empty")

    events: List[AttackEvent] = []
    for i in range(warmup_steps):
        text = warmup_texts[i % len(warmup_texts)]
        events.append(_event(
            start_step + i, "slow_burn", source_id, family_id, text,
            doc_id=f"{source_id}_warmup_{i}", is_poison=False, phase="warmup",
            metadata={"warmup_index": i, "purpose": "build source reputation"},
        ))
    attack_start = start_step + warmup_steps
    for i, payload in enumerate(payloads):
        events.append(_event(
            attack_start + i, "slow_burn", source_id, family_id, payload,
            doc_id=f"{source_id}_poison_{i}",
            target_query=(target_queries[i] if target_queries else None),
            metadata={"warmup_steps": warmup_steps, "phase": "attack"},
        ))
    return events


def framing(
    payloads: Optional[List[str]] = None,
    *,
    targets: Optional[Sequence[Tuple[str, str]]] = None,
    start_step: int = 0,
    source_prefix: str = "framing_attacker",
    family_prefix: str = "framing_family",
    variants_per_target: int = 2,
    target_query: Optional[str] = None,
) -> List[AttackEvent]:
    """Create attacks intended to make legitimate evidence appear isolated.

    Each legitimate ``(doc_id, text)`` target produces ``variants_per_target``
    contradictory passages from *distinct* fresh sources.  The contradiction
    reuses the real claim so the passages stay topical; the point is that the
    legitimate passage now has several independent-looking rivals.
    """
    if variants_per_target < 1:
        raise ValueError("variants_per_target must be >= 1")

    texts: List[Tuple[Optional[str], str]] = []
    if targets:
        for target_doc_id, text in targets:
            first = _first_claim(text)
            texts.append((target_doc_id, _deny(first)))
    for payload in payloads or []:
        texts.append((None, payload))
    if not texts:
        raise ValueError("framing needs payloads or targets")

    events: List[AttackEvent] = []
    for i, (target_doc_id, text) in enumerate(texts):
        for variant in range(variants_per_target):
            source_id = f"{source_prefix}_{i}_{variant}"
            events.append(_event(
                start_step + i * variants_per_target + variant,
                "framing", source_id, f"{family_prefix}_{i}_{variant}", text,
                doc_id=f"{source_id}_doc", target_query=target_query,
                target_doc_id=target_doc_id,
                metadata={"variant": variant, "goal": "isolate legitimate evidence"},
            ))
    return events


_QUESTION_PREFIX = re.compile(
    r"^\s*(who|what|when|where|why|how|which)\b[^?]{0,200}\?\s*", re.IGNORECASE
)


def _remove_query_echo(text: str, query: Optional[str]) -> str:
    """Remove the literal query surface from a payload (signal S1 evasion)."""
    cleaned = text
    if query:
        cleaned = re.sub(re.escape(query.strip()), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(re.escape(query.strip().rstrip("?")), " ", cleaned,
                         flags=re.IGNORECASE)
    cleaned = _QUESTION_PREFIX.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def signal_evasion(
    payloads: List[str],
    start_step: int = 0,
    source_id: str = "evasive_attacker",
    family_id: str = "evasive_family",
    *,
    queries: Optional[Sequence[str]] = None,
    target_queries: Optional[Sequence[str]] = None,
) -> List[AttackEvent]:
    """Construct attacks whose text avoids the literal query echo (S1).

    This is a *measured* evasion claim, not a label: when a query is supplied,
    the emitted payload no longer contains that query string, so bigram/LCS
    query-echo scores collapse.  It does not evade S3 (cluster tightness) or S6
    (neighbourhood duplication) - those need several coordinated sources, which
    is the job of :func:`framing` and :func:`hit_and_run`.
    """
    events: List[AttackEvent] = []
    for i, payload in enumerate(payloads):
        query = queries[i] if queries and i < len(queries) else None
        cleaned = _remove_query_echo(payload, query)
        if not cleaned:
            raise ValueError("signal evasion produced an empty payload")
        if query and query.strip().lower() in cleaned.lower():
            raise AssertionError("signal evasion failed to remove the literal query")
        events.append(_event(
            start_step + i, "signal_evasion", source_id, family_id, cleaned,
            doc_id=f"{source_id}_doc_{i}",
            target_query=(target_queries[i] if target_queries else query),
            metadata={"evaded_signals": ["s1_query_echo"],
                      "not_evaded": ["s3_cluster_tightness", "s6_neighbourhood_density"]},
        ))
    return events


def _first_claim(text: str) -> str:
    """First sentence of a passage, used as the claim a framing attack denies."""
    for separator in (". ", "! ", "? "):
        if separator in text:
            return text.split(separator)[0].strip() + "."
    return text.strip()


def _deny(claim: str) -> str:
    clean = claim.rstrip(".!? ")
    return f"It is not true that {clean}."


def schedule_bounds(events: Iterable[AttackEvent]) -> Tuple[int, int]:
    """(first_step, last_step) of a set of events, for schedule validation."""
    steps = [event.step for event in events]
    if not steps:
        raise ValueError("no events")
    return min(steps), max(steps)


def poison_events(events: Iterable[AttackEvent]) -> List[AttackEvent]:
    """Only the malicious events (warm-up is excluded from attack metrics)."""
    return [event for event in events if event.is_poison]
