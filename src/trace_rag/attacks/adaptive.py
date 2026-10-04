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

    def to_dict(self):
        return {
            "step": int(self.step),
            "attack_type": self.attack_type,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "payload": self.payload,
        }


def hit_and_run(
    payloads: List[str],
    start_step: int = 0,
    source_prefix: str = "attacker",
) -> List[AttackEvent]:
    """Inject each payload from a fresh source to avoid reputation buildup."""
    return [
        AttackEvent(
            step=start_step + i,
            attack_type="hit_and_run",
            source_id=f"{source_prefix}_{i}",
            family_id=f"{source_prefix}_family_{i}",
            payload=payload,
        )
        for i, payload in enumerate(payloads)
    ]


def slow_burn(
    payloads: List[str],
    warmup_steps: int = 5,
    source_id: str = "attacker",
    family_id: str = "attacker_family",
) -> List[AttackEvent]:
    """Build a source's history before injecting malicious payloads."""
    return [
        AttackEvent(
            step=warmup_steps + i,
            attack_type="slow_burn",
            source_id=source_id,
            family_id=family_id,
            payload=payload,
        )
        for i, payload in enumerate(payloads)
    ]


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
