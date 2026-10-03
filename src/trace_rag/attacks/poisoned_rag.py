"""Poisoned-document attack generators for Person C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class PoisonedDocument:
    """A generated poisoned passage plus its attack metadata."""

    doc_id: str
    chunk_id: str
    text: str
    source_id: str
    family_id: str
    attack_type: str
    target_query: Optional[str] = None
    metadata: Mapping[str, object] = None


def entity_swap(text: str, original: str, replacement: str) -> str:
    """Replace a target entity with an attacker-chosen entity."""
    if not original:
        raise ValueError("original entity must not be empty")
    return text.replace(original, replacement)


def negation(text: str, claim: str) -> str:
    """Create a simple negation-based corruption of a claim."""
    if not claim:
        raise ValueError("claim must not be empty")

    clean_claim = claim.rstrip(".!? ")
    replacement = f"It is not true that {clean_claim}."
    if claim in text:
        return text.replace(claim, replacement, 1)

    return f"Not true: {claim}. {text}"


def instruction_injection(text: str, instruction: str) -> str:
    """Append an instruction payload without echoing the user's query."""
    if not instruction:
        raise ValueError("instruction must not be empty")

    return (
        f"{text}\n\n"
        "[DOCUMENT INSTRUCTION]\n"
        f"{instruction}\n"
        "[END DOCUMENT INSTRUCTION]"
    )


def make_poisoned_document(
    *,
    doc_id: str,
    chunk_id: str,
    text: str,
    source_id: str,
    family_id: str,
    attack_type: str,
    target_query: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> PoisonedDocument:
    """Construct a poison record for insertion into an evaluation corpus."""
    if not doc_id or not chunk_id:
        raise ValueError("doc_id and chunk_id must not be empty")
    if not text:
        raise ValueError("text must not be empty")

    return PoisonedDocument(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        source_id=source_id,
        family_id=family_id,
        attack_type=attack_type,
        target_query=target_query,
        metadata=dict(metadata or {}),
    )
