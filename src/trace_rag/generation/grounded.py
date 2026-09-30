"""Grounded generation with mandatory citations and evidence-based abstention.

The rule (plan Section 4.9): answer only from the passages that survived
policy, cite every sentence, and abstain when the trust-weighted mass of cited
evidence is below tau_ans.  Abstaining is a *result*, not a failure: it is what
keeps the system from repeating a poisoned claim it cannot corroborate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..config import GenerationConfig
from ..contracts import AnswerRecord, Citation, LLMResponse, RetrievedDocument
from ..utils.timing import Stopwatch
from .citations import check_citations, citation_precision, evidence_mass, parse_citations
from .prompts import build_answer_prompt

ABSTAIN_TEXT = "INSUFFICIENT EVIDENCE"
ABSTAIN_MESSAGE = (
    "I don't have enough trusted evidence in the knowledge base to answer this reliably."
)


@dataclass
class GenerationOutcome:
    record: AnswerRecord
    raw_response: Optional[LLMResponse]
    prompt: str
    invalid_citations: Tuple[str, ...] = ()

    @property
    def answer(self) -> str:
        return self.record.answer


class GroundedGenerator:
    def __init__(self, llm, config: Optional[GenerationConfig] = None,  # type: ignore[no-untyped-def]
                 citation_checker: Optional[Callable[[str, str], Tuple[bool, float]]] = None) -> None:
        self.llm = llm
        self.config = config or GenerationConfig()
        self.citation_checker = citation_checker

    def generate(self, query: str, query_id: str, documents: Sequence[RetrievedDocument],
                 excluded_doc_ids: Sequence[str] = (), step: int = 0) -> GenerationOutcome:
        documents = list(documents)[: self.config.max_context_docs]
        trust_snapshots = {d.doc_id: d.trust.to_dict() for d in documents}

        if not documents:
            return self._abstain(query, query_id, "no_context", documents, excluded_doc_ids,
                                 trust_snapshots, step, prompt="", response=None)

        prompt = build_answer_prompt(query, documents)
        with Stopwatch() as watch:
            response = self.llm.generate(prompt, max_tokens=self.config.max_tokens)

        text = (response.text or "").strip()
        if not text or ABSTAIN_TEXT.lower() in text.lower():
            return self._abstain(query, query_id, "model_abstained", documents, excluded_doc_ids,
                                 trust_snapshots, step, prompt, response, watch.elapsed_ms)

        citations, invalid, cleaned = parse_citations(text, documents)
        citations = check_citations(citations, documents, self.citation_checker)

        if self.config.require_citations and not citations:
            reason = "invalid_citations" if invalid else "no_citations"
            return self._abstain(query, query_id, reason, documents, excluded_doc_ids,
                                 trust_snapshots, step, prompt, response, watch.elapsed_ms,
                                 invalid=invalid)

        mass = evidence_mass(citations, documents)
        if mass < self.config.abstain_evidence_mass:
            return self._abstain(query, query_id, "insufficient_evidence", documents,
                                 excluded_doc_ids, trust_snapshots, step, prompt, response,
                                 watch.elapsed_ms, invalid=invalid, mass=mass,
                                 citations=tuple(citations))

        if self.citation_checker is not None:
            precision = citation_precision(citations)
            if precision is not None and precision == 0.0:
                return self._abstain(query, query_id, "citations_unsupported", documents,
                                     excluded_doc_ids, trust_snapshots, step, prompt, response,
                                     watch.elapsed_ms, invalid=invalid, mass=mass,
                                     citations=tuple(citations))

        record = AnswerRecord(
            query_id=query_id, query=query, answer=text, abstained=False, abstain_reason=None,
            citations=tuple(citations),
            used_doc_ids=tuple(sorted({c.doc_id for c in citations})),
            excluded_doc_ids=tuple(excluded_doc_ids), trust_snapshots=trust_snapshots,
            evidence_mass=mass, llm_calls=int(response.llm_calls),
            latency_ms=float(watch.elapsed_ms), step=step, model=getattr(response, "model", "unknown"),
        )
        return GenerationOutcome(record=record, raw_response=response, prompt=prompt,
                                 invalid_citations=tuple(invalid))

    # ----------------------------------------------------------------- helper
    def _abstain(self, query: str, query_id: str, reason: str,
                 documents: Sequence[RetrievedDocument], excluded: Sequence[str],
                 trust_snapshots: Dict[str, Dict[str, object]], step: int, prompt: str,
                 response: Optional[LLMResponse], latency_ms: float = 0.0,
                 invalid: Sequence[str] = (), mass: float = 0.0,
                 citations: Tuple[Citation, ...] = ()) -> GenerationOutcome:
        record = AnswerRecord(
            query_id=query_id, query=query, answer=ABSTAIN_MESSAGE, abstained=True,
            abstain_reason=reason, citations=citations,
            used_doc_ids=tuple(sorted({c.doc_id for c in citations})),
            excluded_doc_ids=tuple(excluded), trust_snapshots=trust_snapshots,
            evidence_mass=float(mass), llm_calls=int(response.llm_calls) if response else 0,
            latency_ms=float(latency_ms), step=step,
            model=getattr(response, "model", "none") if response else "none",
        )
        return GenerationOutcome(record=record, raw_response=response, prompt=prompt,
                                 invalid_citations=tuple(invalid))
