"""Grounded generation with mandatory citations and evidence-based abstention.

The rule (plan Section 4.9): answer only from the passages that survived
policy, cite every sentence, and abstain when the trust-weighted mass of cited
evidence is below tau_ans.  Abstaining is a *result*, not a failure: it is what
keeps the system from repeating a poisoned claim it cannot corroborate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from ..config import GenerationConfig
from ..contracts import AnswerRecord, Citation, LLMResponse, RetrievedDocument, TrustStatus
from ..utils.logging import get_logger
from ..utils.timing import Stopwatch
from .citations import check_citations, citation_precision, evidence_mass, parse_citations
from .llm import generate_with_messages
from .prompts import build_answer_messages, messages_to_prompt

logger = get_logger(__name__)

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
                 excluded_doc_ids: Sequence[str] = (), step: int = 0,
                 now: Optional[float] = None) -> GenerationOutcome:
        # Defence in depth: policy already removes blocked passages, but a
        # quarantined document must never reach the generator even if a caller
        # assembled the context list itself.
        blocked = [d.doc_id for d in documents
                   if d.trust.status in (TrustStatus.QUARANTINED, TrustStatus.REJECTED)]
        documents = [d for d in documents if d.trust.status not in
                     (TrustStatus.QUARANTINED, TrustStatus.REJECTED)]
        documents = list(documents)[: self.config.max_context_docs]
        excluded = tuple(dict.fromkeys(list(excluded_doc_ids) + blocked))
        trust_snapshots = {d.doc_id: d.trust.to_dict() for d in documents}
        timestamp = time.time() if now is None else float(now)

        if not documents:
            return self._abstain(query, query_id, "no_context", documents, excluded,
                                 trust_snapshots, step, prompt="", response=None,
                                 timestamp=timestamp)

        messages = build_answer_messages(query, documents)
        prompt = messages_to_prompt(messages)
        with Stopwatch() as watch:
            try:
                response = generate_with_messages(self.llm, messages,
                                                  max_tokens=self.config.max_tokens)
            except Exception as exc:  # noqa: BLE001 - fail closed, keep the attempt
                # A generation outage is a result, not a reason to abort the
                # whole stream: the query remains an eligible attempt and is
                # recorded as an abstention rather than silently disappearing
                # from the denominator.
                logger.warning("generation backend failed for %s: %s: %s",
                               query_id, type(exc).__name__, exc)
                return self._abstain(query, query_id, "generation_error", documents,
                                     excluded, trust_snapshots, step, prompt,
                                     response=None, latency_ms=watch.elapsed_ms,
                                     timestamp=timestamp)

        text = (response.text or "").strip()
        if not text or ABSTAIN_TEXT.lower() in text.lower():
            return self._abstain(query, query_id, "model_abstained", documents, excluded,
                                 trust_snapshots, step, prompt, response, watch.elapsed_ms,
                                 timestamp=timestamp)

        citations, invalid, cleaned = parse_citations(text, documents)
        citations = check_citations(citations, documents, self.citation_checker)

        if self.config.require_citations and not citations:
            reason = "invalid_citations" if invalid else "no_citations"
            return self._abstain(query, query_id, reason, documents, excluded,
                                 trust_snapshots, step, prompt, response, watch.elapsed_ms,
                                 invalid=invalid, timestamp=timestamp)

        mass = evidence_mass(citations, documents)
        if mass < self.config.abstain_evidence_mass:
            return self._abstain(query, query_id, "insufficient_evidence", documents,
                                 excluded, trust_snapshots, step, prompt, response,
                                 watch.elapsed_ms, invalid=invalid, mass=mass,
                                 citations=tuple(citations), timestamp=timestamp)

        if self.citation_checker is not None:
            precision = citation_precision(citations)
            if precision is not None and precision == 0.0:
                return self._abstain(query, query_id, "citations_unsupported", documents,
                                     excluded, trust_snapshots, step, prompt, response,
                                     watch.elapsed_ms, invalid=invalid, mass=mass,
                                     citations=tuple(citations), timestamp=timestamp)

        record = AnswerRecord(
            query_id=query_id, query=query, answer=text, abstained=False, abstain_reason=None,
            citations=tuple(citations),
            used_doc_ids=tuple(sorted({c.doc_id for c in citations})),
            excluded_doc_ids=excluded, trust_snapshots=trust_snapshots,
            evidence_mass=mass, llm_calls=int(response.llm_calls),
            latency_ms=float(watch.elapsed_ms), step=step, timestamp=timestamp,
            model=getattr(response, "model", "unknown"),
        )
        return GenerationOutcome(record=record, raw_response=response, prompt=prompt,
                                 invalid_citations=tuple(invalid))

    # ----------------------------------------------------------------- helper
    def _abstain(self, query: str, query_id: str, reason: str,
                 documents: Sequence[RetrievedDocument], excluded: Sequence[str],
                 trust_snapshots: Dict[str, Dict[str, object]], step: int, prompt: str,
                 response: Optional[LLMResponse], latency_ms: float = 0.0,
                 invalid: Sequence[str] = (), mass: float = 0.0,
                 citations: Tuple[Citation, ...] = (),
                 timestamp: Optional[float] = None) -> GenerationOutcome:
        record = AnswerRecord(
            query_id=query_id, query=query, answer=ABSTAIN_MESSAGE, abstained=True,
            abstain_reason=reason, citations=citations,
            used_doc_ids=tuple(sorted({c.doc_id for c in citations})),
            excluded_doc_ids=tuple(excluded), trust_snapshots=trust_snapshots,
            evidence_mass=float(mass), llm_calls=int(response.llm_calls) if response else 0,
            latency_ms=float(latency_ms), step=step,
            timestamp=time.time() if timestamp is None else float(timestamp),
            model=getattr(response, "model", "none") if response else "none",
        )
        return GenerationOutcome(record=record, raw_response=response, prompt=prompt,
                                 invalid_citations=tuple(invalid))
