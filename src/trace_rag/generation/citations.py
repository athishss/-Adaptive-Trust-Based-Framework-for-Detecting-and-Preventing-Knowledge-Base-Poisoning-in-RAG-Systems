"""Citation parsing and trust-weighted evidence mass."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import Citation, RetrievedDocument
from ..utils.textnorm import sentences

CITATION_PATTERN = re.compile(r"\[([^\[\]]{1,120}?)\]")


def parse_citations(answer: str, allowed: Sequence[RetrievedDocument]
                    ) -> Tuple[List[Citation], List[str], str]:
    """Extract per-sentence citations.

    Returns (citations, invalid_ids, cleaned_answer).  Identifiers the model
    invented are reported rather than silently dropped: hallucinated citations
    are a measurable failure mode, and Person C reports them.
    """
    allowed_map: Dict[str, RetrievedDocument] = {d.doc_id: d for d in allowed}
    citations: List[Citation] = []
    invalid: List[str] = []
    for sentence in sentences(answer):
        for raw in CITATION_PATTERN.findall(sentence):
            for candidate in (part.strip() for part in raw.split(",")):
                if not candidate:
                    continue
                document = allowed_map.get(candidate)
                if document is None:
                    invalid.append(candidate)
                    continue
                citations.append(Citation(doc_id=document.doc_id, chunk_id=document.chunk_id,
                                          sentence=CITATION_PATTERN.sub("", sentence).strip()))
    cleaned = CITATION_PATTERN.sub("", answer)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)          # leftover empty brackets
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return citations, invalid, cleaned


def evidence_mass(citations: Sequence[Citation], documents: Sequence[RetrievedDocument]) -> float:
    """Trust-weighted mass of the cited evidence, aggregated by *source*.

    Independent sources compound (noisy-OR), passages from the same source do
    not.  Five agreeing passages uploaded by one contributor therefore count
    once, which is the same independence rule Person B's verifier applies to
    corroboration.  Range [0, 1].
    """
    by_id = {d.doc_id: d for d in documents}
    best_per_source: Dict[str, float] = {}
    for citation in citations:
        document = by_id.get(citation.doc_id)
        if document is None:
            continue
        trust = float(document.trust.t_eff)
        source = document.source_id
        best_per_source[source] = max(best_per_source.get(source, 0.0), trust)
    if not best_per_source:
        return 0.0
    product = 1.0
    for trust in best_per_source.values():
        product *= (1.0 - min(max(trust, 0.0), 1.0))
    return float(1.0 - product)


def check_citations(citations: Sequence[Citation], documents: Sequence[RetrievedDocument],
                    checker: Optional[Callable[[str, str], Tuple[bool, float]]] = None
                    ) -> List[Citation]:
    """Optionally verify each cited sentence against its passage.

    ``checker(premise, hypothesis) -> (supported, score)`` is Person B's NLI
    cross-encoder.  Without it the citations are returned unchanged with
    ``supported=None``, never silently marked as verified.
    """
    if checker is None:
        return list(citations)
    by_id = {d.doc_id: d for d in documents}
    checked: List[Citation] = []
    for citation in citations:
        document = by_id.get(citation.doc_id)
        if document is None or not citation.sentence:
            checked.append(citation)
            continue
        supported, score = checker(document.text, citation.sentence)
        checked.append(Citation(doc_id=citation.doc_id, chunk_id=citation.chunk_id,
                                sentence=citation.sentence, supported=bool(supported),
                                nli_score=float(score)))
    return checked


def citation_precision(citations: Sequence[Citation]) -> Optional[float]:
    checked = [c for c in citations if c.supported is not None]
    if not checked:
        return None
    return float(sum(1 for c in checked if c.supported) / len(checked))
