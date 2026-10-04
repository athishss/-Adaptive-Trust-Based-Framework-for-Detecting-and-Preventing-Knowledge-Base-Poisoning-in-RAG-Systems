"""Poisoned-document attack generators for Person C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "attack_type": self.attack_type,
            "target_query": self.target_query,
            "metadata": dict(self.metadata),
        }


def entity_swap(text: str, original: str, replacement: str) -> str:
    """Replace every occurrence of an entity with the attacker-chosen entity."""
    if not original:
        raise ValueError("original entity must not be empty")
    if not replacement:
        raise ValueError("replacement entity must not be empty")
    if original not in text:
        raise ValueError("original entity was not found in text")

    return text.replace(original, replacement)


def negation(text: str, claim: str) -> str:
    """Explicitly negate a target claim inside a document."""
    if not claim:
        raise ValueError("claim must not be empty")

    clean_claim = claim.rstrip(".!? ")
    replacement = f"It is not true that {clean_claim}."

    if claim in text:
        return text.replace(claim, replacement, 1)

    return f"{replacement} {text}"


def instruction_injection(text: str, instruction: str) -> str:
    """Append a document-level instruction payload."""
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
    metadata: Optional[Mapping[str, Any]] = None,
) -> PoisonedDocument:
    """Construct a poison record for insertion into an evaluation corpus."""
    if not doc_id or not chunk_id:
        raise ValueError("doc_id and chunk_id must not be empty")
    if not text:
        raise ValueError("text must not be empty")
    if not source_id:
        raise ValueError("source_id must not be empty")
    if not family_id:
        raise ValueError("family_id must not be empty")
    if not attack_type:
        raise ValueError("attack_type must not be empty")

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
