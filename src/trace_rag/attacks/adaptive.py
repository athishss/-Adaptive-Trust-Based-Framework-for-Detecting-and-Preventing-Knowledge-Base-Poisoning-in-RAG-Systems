"""Adaptive trust-aware attack schedules for Person C."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


@dataclass(frozen=True)
class AttackEvent:
    """One attack or warm-up ingestion event in chronological order."""

    step: int
    attack_type: str
    source_id: str
    family_id: str
    payload: str
    is_poison: bool = True

    def to_dict(self) -> dict:
        return {
            "step": int(self.step),
            "attack_type": self.attack_type,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "payload": self.payload,
            "is_poison": bool(self.is_poison),
        }


def hit_and_run(payloads: List[str], start_step: int = 0,
                source_prefix: str = "attacker") -> List[AttackEvent]:
    """Schedule each supplied poison from a fresh source/family identity."""
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if any(not payload.strip() for payload in payloads):
        raise ValueError("payloads must not contain empty text")
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


def slow_burn(payloads: List[str], warmup_steps: int = 5,
              source_id: str = "attacker", family_id: str = "attacker_family",
              warmup_payloads: Optional[List[str]] = None,
              start_step: int = 0) -> List[AttackEvent]:
    """Schedule real benign history before malicious passages from one source.

    Each warm-up item must be caller-supplied clean content; synthesizing
    arbitrary "benign" prose would not build legitimate verifier support. The
    attack phase starts only after all requested warm-up events have appeared.
    """
    if warmup_steps < 0 or start_step < 0:
        raise ValueError("warmup_steps and start_step must be non-negative")
    warmups = list(warmup_payloads or ())
    if warmup_steps and len(warmups) < warmup_steps:
        raise ValueError("slow_burn requires one clean warmup_payload per warmup step")
    if any(not payload.strip() for payload in [*warmups[:warmup_steps], *payloads]):
        raise ValueError("warm-up and poison payloads must not contain empty text")

    events = [
        AttackEvent(
            step=start_step + i,
            attack_type="slow_burn_warmup",
            source_id=source_id,
            family_id=family_id,
            payload=warmups[i],
            is_poison=False,
        )
        for i in range(warmup_steps)
    ]
    events.extend(
        AttackEvent(
            step=start_step + warmup_steps + i,
            attack_type="slow_burn",
            source_id=source_id,
            family_id=family_id,
            payload=payload,
            is_poison=True,
        )
        for i, payload in enumerate(payloads)
    )
    return events


def framing(payloads: List[str], start_step: int = 0,
            source_id: str = "framing_attacker",
            family_id: str = "framing_family") -> List[AttackEvent]:
    """Schedule caller-crafted framing payloads as one shared-source burst.

    The actual framing construction (which clean/poison passages to surround)
    is dataset-specific and stays with the experiment builder; this helper
    makes the shared identity and synchronized attack timing explicit.
    """
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if any(not payload.strip() for payload in payloads):
        raise ValueError("payloads must not contain empty text")
    return [
        AttackEvent(start_step + i, "framing", source_id, family_id, payload)
        for i, payload in enumerate(payloads)
    ]


def _remove_question_echo(payload: str) -> str:
    """Remove a leading literal question prefix used by query-echo attacks."""
    # PoisonedRAG's query-concatenation format places the whole question before
    # the injected claim. Strip that prefix when it is recognizable, leaving
    # the malicious claim intact for an S1-evasion ablation.
    return re.sub(r"^\s*[^?\n]{1,240}\?\s*", "", payload, count=1).strip()


def signal_evasion(payloads: List[str], start_step: int = 0,
                   source_id: str = "evasive_attacker",
                   family_id: str = "evasive_family") -> List[AttackEvent]:
    """Strip obvious leading query echoes, then schedule S1-evasion payloads."""
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    events: List[AttackEvent] = []
    for i, payload in enumerate(payloads):
        if not payload.strip():
            raise ValueError("payloads must not contain empty text")
        transformed = _remove_question_echo(payload) or payload.strip()
        events.append(AttackEvent(
            step=start_step + i,
            attack_type="signal_evasion",
            source_id=source_id,
            family_id=family_id,
            payload=transformed,
        ))
    return events
