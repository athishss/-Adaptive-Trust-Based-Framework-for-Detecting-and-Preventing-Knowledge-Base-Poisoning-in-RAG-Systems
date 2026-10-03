"""Adaptive trust-aware attack schedules for Person C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AttackEvent:
    """One attack injection event in chronological order."""

    step: int
    attack_type: str
    source_id: str
    family_id: str
    payload: str


def hit_and_run(
    payloads: List[str],
    start_step: int = 0,
    source_prefix: str = "attacker",
) -> List[AttackEvent]:
    """Inject each payload from a fresh source to avoid reputation buildup."""
    events: List[AttackEvent] = []

    for i, payload in enumerate(payloads):
        events.append(
            AttackEvent(
                step=start_step + i,
                attack_type="hit_and_run",
                source_id=f"{source_prefix}_{i}",
                family_id=f"{source_prefix}_family_{i}",
                payload=payload,
            )
        )

    return events


def slow_burn(
    payloads: List[str],
    warmup_steps: int = 5,
    source_id: str = "attacker",
    family_id: str = "attacker_family",
) -> List[AttackEvent]:
    """Build a source's history before injecting the malicious payloads."""
    events: List[AttackEvent] = []

    for i, payload in enumerate(payloads):
        events.append(
            AttackEvent(
                step=warmup_steps + i,
                attack_type="slow_burn",
                source_id=source_id,
                family_id=family_id,
                payload=payload,
            )
        )

    return events


def framing(
    payloads: List[str],
    start_step: int = 0,
    source_id: str = "framing_attacker",
    family_id: str = "framing_family",
) -> List[AttackEvent]:
    """Create attacks intended to make legitimate evidence appear isolated."""
    return [
        AttackEvent(
            step=start_step + i,
            attack_type="framing",
            source_id=source_id,
            family_id=family_id,
            payload=payload,
        )
        for i, payload in enumerate(payloads)
    ]


def signal_evasion(
    payloads: List[str],
    start_step: int = 0,
    source_id: str = "evasive_attacker",
    family_id: str = "evasive_family",
) -> List[AttackEvent]:
    """Create attacks whose text avoids obvious query-echo patterns."""
    return [
        AttackEvent(
            step=start_step + i,
            attack_type="signal_evasion",
            source_id=source_id,
            family_id=family_id,
            payload=payload,
        )
        for i, payload in enumerate(payloads)
    ]