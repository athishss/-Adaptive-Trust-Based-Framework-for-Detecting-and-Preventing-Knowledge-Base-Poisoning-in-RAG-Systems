"""Poisoned-document attack generators for Person C."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
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
    match = re.search(rf"(?<!\w){re.escape(original)}(?!\w)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("original entity was not found in text")
    # A single targeted swap models one corrupted claim; replacing every
    # mention can create unnatural contradictory passages and leak attack labels.
    return text[:match.start()] + replacement + text[match.end():]


def negation(text: str, claim: str) -> str:
    """Replace an affirmative claim with its negation, preserving surrounding text.

    Matching is case-insensitive and ignores terminal punctuation. If the exact
    claim phrase is absent, the negated claim is prepended as an injected
    sentence; the helper never silently leaves an exact affirmative copy in
    place when it can replace it.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must not be empty")

    phrase = claim.strip().rstrip(".!? ")
    if not phrase:
        raise ValueError("claim must contain non-punctuation text")
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)[.!?]?", re.IGNORECASE)
    match = pattern.search(text)
    found = match.group(0) if match is not None else phrase
    clean_claim = found.rstrip(".!? ")
    replacement = f"It is not true that {clean_claim}."
    if match is not None:
        return text[:match.start()] + replacement + text[match.end():]
    return f"{replacement} {text}".strip()


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
