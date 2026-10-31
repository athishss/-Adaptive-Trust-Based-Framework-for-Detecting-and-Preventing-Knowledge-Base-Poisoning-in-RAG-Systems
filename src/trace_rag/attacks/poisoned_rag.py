"""Poisoned-document attack generators for Person C.

Every generator here changes the corpus in a defined way.  The earlier versions
of ``entity_swap`` (unbounded replacement) and ``negation`` (prepend a denial
while keeping the true claim) produced passages that did not correspond to the
attack they were named after; the specs below are the contract the tests hold
them to.

=============  ===============================  ================================
attack         transformation                   insertion point / target
=============  ===============================  ================================
entity_swap    one targeted entity/value         an ingested passage that answers
               replacement (default: first       a target query; family/source
               occurrence, word-boundary,        belong to the attacker
               case-insensitive)
negation       replace the target claim with a   same passage, same retrieval
               sentence asserting its negation   surface
instruction_   append a delimited instruction    any retrieved passage; the
injection      payload                           attack succeeds if the model
                                                 follows it, not if it is merely
                                                 present
=============  ===============================  ================================

Ground-truth labels are attached by the caller (``is_poison`` / ``attack_type``);
the generators never decide what counts as a successful attack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

ATTACK_SPECS: Dict[str, Dict[str, Any]] = {
    "entity_swap": {
        "assumptions": ["attacker knows the target entity/value",
                        "attacker can ingest a passage that answers the query"],
        "transform": "replace one occurrence of the true entity with a false one",
        "target": "a passage retrieved for the target query",
        "expected": ["passage keeps the query context", "false entity is asserted once"],
        "label": "is_poison=True, attack_type='entity_swap'",
    },
    "negation": {
        "assumptions": ["attacker knows the claim asserted by the target passage"],
        "transform": "replace the claim with 'It is not true that <claim>.'",
        "target": "the sentence carrying the answer",
        "expected": ["the original assertion is gone", "surrounding context is retained"],
        "label": "is_poison=True, attack_type='negation'",
    },
    "instruction_injection": {
        "assumptions": ["attacker can place arbitrary text in a retrieved passage"],
        "transform": "append a delimited instruction payload",
        "target": "any passage retrieved for the target query",
        "expected": ["payload is present", "a secure generator must not follow it"],
        "label": "is_poison=True, attack_type='instruction_injection'",
    },
}


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


def entity_swap(text: str, original: str, replacement: str, *,
                count: int = 1, case_sensitive: bool = False) -> str:
    """Replace *at most* ``count`` occurrences of a target entity.

    PoisonedRAG's construction needs one targeted replacement, not a global
    rewrite: replacing every occurrence changes unrelated sentences and can
    destroy the retrieval surface that made the passage relevant.  Matching is
    word-boundary anchored so ``Eiffel`` does not rewrite ``Eiffel Tower``'s
    neighbours and ``Paris`` does not rewrite ``Parisian``.
    """
    if not original:
        raise ValueError("original entity must not be empty")
    if not replacement:
        raise ValueError("replacement entity must not be empty")
    if count < 1:
        raise ValueError("count must be >= 1")

    pattern = re.compile(rf"\b{re.escape(original)}\b",
                         0 if case_sensitive else re.IGNORECASE)
    swapped, replacements = pattern.subn(lambda _: replacement, text, count=count)
    if replacements == 0:
        raise ValueError(f"original entity {original!r} was not found in text")
    return swapped


def negation(text: str, claim: str, *, replacement: Optional[str] = None) -> str:
    """Replace a claim with an explicit denial of it.

    The true claim must be removed, not merely prefixed with a denial: keeping
    it lets a corroboration verifier read the retained sentence as support and
    turns the "corruption" into a self-contradictory document.  If the claim is
    not present, this raises instead of silently producing a benign passage.
    """
    if not claim:
        raise ValueError("claim must not be empty")
    if claim not in text:
        raise ValueError(
            "claim was not found verbatim in the passage; refusing to emit a "
            "passage that still asserts the true claim"
        )

    clean_claim = claim.rstrip(".!? ")
    denial = replacement or f"It is not true that {clean_claim}."
    return text.replace(claim, denial, 1)


def instruction_injection(text: str, instruction: str) -> str:
    """Append a document-level instruction payload.

    The payload is data.  Application logic must keep it out of the system role
    and must not follow it; this generator only produces the adversarial input.
    """
    if not instruction:
        raise ValueError("instruction must not be empty")
    if "</document" in instruction.lower():
        raise ValueError("instruction must not contain a closing document tag")
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
