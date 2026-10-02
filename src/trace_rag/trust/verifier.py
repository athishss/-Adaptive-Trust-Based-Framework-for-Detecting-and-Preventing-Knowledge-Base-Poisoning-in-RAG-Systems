"""Corroboration-gated counterfactual verifier (plan Section 4.5).

Called only for MEDIUM-band passages.  The algorithm:

1. **Single-document claim extraction.**  Ask the LLM "what does this passage
   alone say about the query?" using Person A's ``build_single_doc_prompt``.

2. **Find independent sources.**  From the candidate pool, keep only passages
   whose ``source_id`` *and* ``family_id`` both differ from the target's.
   This is what makes corroboration meaningful — five passages uploaded by the
   same attacker cannot corroborate each other.

3. **Corroboration check.**  For each independent passage, extract its claim
   about the same question and compare it to the target's claim lexically.

4. **Aggregate.**  Trust-weighted support and refute masses (noisy-OR by
   source, so passages from the same source count once).

5. **Decide.**  SUPPORT if the balance favours agreement with independent
   sources; REFUTE if independent sources contradict; NEUTRAL otherwise.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..contracts import RetrievedDocument, VerificationOutcome, VerificationResult
from ..generation.llm import StubLLM
from ..generation.prompts import build_single_doc_prompt
from ..utils.textnorm import tokenise
from ..utils.timing import Stopwatch
from ..utils.logging import get_logger

# Common English function words that dilute Jaccard overlap.
# Stripping these lets content words (names, nouns, verbs) drive the score.
_STOP_WORDS = frozenset(
    "a an the is was were are be been being am "
    "in on at to by for of from with as into "
    "and or but not no nor so yet "
    "it its he she they them his her their "
    "this that these those "
    "has had have do does did will would shall should "
    "can could may might must "
    "than very also just".split()
)

logger = get_logger(__name__)

# Matches "[anything]" citation markers in LLM output
_CITATION_RE = re.compile(r"\[[^\[\]]{1,120}?\]")

# Marker the StubLLM returns when it cannot answer
_ABSTAIN_MARKER = "INSUFFICIENT EVIDENCE"


class CorroborationVerifier:
    """Corroboration-gated counterfactual verifier.

    Satisfies the ``Verifier`` protocol from ``contracts.py``.

    Parameters
    ----------
    llm : LLM-like
        Language model used for single-document claim extraction.
        Defaults to ``StubLLM`` (offline, no GPU needed).
    max_corroboration : int
        Maximum number of independent passages to check per verification.
    support_threshold : float
        Minimum claim overlap to count as support evidence.
    refute_threshold : float
        Maximum claim overlap (on a relevant passage) to count as refute.
    min_mass : float
        Minimum aggregated mass required to issue a non-NEUTRAL verdict.
    """

    def __init__(self, llm=None, max_corroboration: int = 3,
                 support_threshold: float = 0.15, refute_threshold: float = 0.10,
                 min_mass: float = 0.10) -> None:
        self.llm = llm or StubLLM()
        self.max_corroboration = int(max_corroboration)
        self.support_threshold = float(support_threshold)
        self.refute_threshold = float(refute_threshold)
        self.min_mass = float(min_mass)

    # ------------------------------------------------------------------
    #  Verifier protocol
    # ------------------------------------------------------------------

    def verify(self, query: str, query_id: str,
               target: RetrievedDocument,
               pool: Sequence[RetrievedDocument]) -> VerificationResult:
        """Verify a MEDIUM-band passage by checking corroboration.

        Returns a ``VerificationResult`` with honest ``llm_calls`` count.
        """
        with Stopwatch() as watch:
            # Step 1 — what does the target passage alone claim?
            target_claim, llm_calls = self._extract_claim(query, target)
            influential = bool(
                target_claim
                and _ABSTAIN_MARKER.lower() not in target_claim.lower()
            )

            # Step 2 — find independent sources in the pool
            independent = self._find_independent(target, pool)

            if not independent or not influential:
                # No independent sources or no substantive claim → NEUTRAL
                return VerificationResult(
                    doc_id=target.doc_id, query_id=query_id,
                    single_doc_answer=target_claim,
                    influential=influential,
                    support_mass=0.0, refute_mass=0.0,
                    outcome=VerificationOutcome.NEUTRAL,
                    llm_calls=llm_calls, latency_ms=watch.elapsed_ms,
                )

            # Step 3 — extract claims from independent passages and compare
            support_mass, refute_mass, extra_calls = self._corroborate(
                query, target_claim, independent,
            )
            llm_calls += extra_calls

            # Step 4 — determine outcome
            outcome = self._decide(support_mass, refute_mass)
            logger.debug(
                "verify %s: support=%.3f refute=%.3f → %s (llm_calls=%d)",
                target.doc_id, support_mass, refute_mass, outcome.value, llm_calls,
            )

        return VerificationResult(
            doc_id=target.doc_id, query_id=query_id,
            single_doc_answer=target_claim,
            influential=influential,
            support_mass=float(support_mass),
            refute_mass=float(refute_mass),
            outcome=outcome,
            llm_calls=int(llm_calls),
            latency_ms=float(watch.elapsed_ms),
        )

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _extract_claim(self, query: str, doc: RetrievedDocument) -> Tuple[str, int]:
        """Ask the LLM what a single passage claims.  Returns (claim_text, llm_calls)."""
        prompt = build_single_doc_prompt(query, doc)
        response = self.llm.generate(prompt, max_tokens=256)
        # Strip citation markers from the answer
        text = _CITATION_RE.sub("", response.text or "").strip()
        return text, int(response.llm_calls)

    def _find_independent(self, target: RetrievedDocument,
                          pool: Sequence[RetrievedDocument]) -> List[RetrievedDocument]:
        """Keep only pool passages from genuinely independent sources.

        Independent means different ``source_id`` *and* different ``family_id``
        from the target (plan Section 4.5, rule 2 in HANDOVER.md).
        """
        candidates = [
            p for p in pool
            if p.doc_id != target.doc_id
            and p.source_id != target.source_id
            and p.family_id != target.family_id
        ]
        # Prioritise the most trusted independent passages
        candidates.sort(key=lambda p: -p.trust.t_eff)
        return candidates[: self.max_corroboration]

    def _corroborate(self, query: str, target_claim: str,
                     independent: List[RetrievedDocument]
                     ) -> Tuple[float, float, int]:
        """Compare target's claim against independent sources.

        Uses a two-pronged comparison:
          1. claim-vs-claim: extracted answer overlap
          2. claim-vs-passage: does the independent passage's *full text*
             support or contradict the target's extracted claim?

        The higher of the two signals is used per passage.

        Returns (support_mass, refute_mass, llm_calls).
        Mass is trust-weighted, aggregated by source (noisy-OR).
        """
        # Extract claims from independent passages
        claims: List[Tuple[RetrievedDocument, str]] = []
        total_llm = 0
        for doc in independent:
            claim, calls = self._extract_claim(query, doc)
            total_llm += calls
            if claim and _ABSTAIN_MARKER.lower() not in claim.lower():
                claims.append((doc, claim))

        if not claims:
            return 0.0, 0.0, total_llm

        # Build content-word sets for the target's claim
        query_tokens = set(tokenise(query)) | _STOP_WORDS
        target_content = set(tokenise(target_claim)) - query_tokens

        best_support_per_source: Dict[str, float] = {}
        best_refute_per_source: Dict[str, float] = {}

        for doc, ind_claim in claims:
            # Compare on two axes and take the stronger signal
            claim_agree = self._content_overlap(target_content, ind_claim, query_tokens)
            passage_agree = self._content_overlap(target_content, doc.text, query_tokens)
            agreement = max(claim_agree, passage_agree)

            trust = float(doc.trust.t_eff)
            source = doc.source_id

            if agreement >= self.support_threshold:
                weight = trust * agreement
                best_support_per_source[source] = max(
                    best_support_per_source.get(source, 0.0), weight,
                )
            elif agreement <= self.refute_threshold:
                weight = trust * (1.0 - agreement)
                best_refute_per_source[source] = max(
                    best_refute_per_source.get(source, 0.0), weight,
                )

        # Noisy-OR aggregation (same rule as evidence_mass in citations.py)
        support_mass = self._noisy_or(best_support_per_source)
        refute_mass = self._noisy_or(best_refute_per_source)
        return support_mass, refute_mass, total_llm

    @staticmethod
    def _content_overlap(target_content: set, text: str,
                         noise: set) -> float:
        """Content-word Jaccard overlap between pre-tokenised *target_content*
        words and the content words of *text*, with *noise* (query + stop words)
        stripped from both sides.

        Returns 0..1: high means agreement, low means disagreement.
        """
        other_content = set(tokenise(text)) - noise
        if not target_content or not other_content:
            return 0.0
        overlap = len(target_content & other_content)
        union = len(target_content | other_content)
        return float(overlap / union) if union else 0.0

    @staticmethod
    def _noisy_or(per_source: Dict[str, float]) -> float:
        """Noisy-OR: 1 − ∏(1 − weight) across independent sources."""
        if not per_source:
            return 0.0
        product = 1.0
        for w in per_source.values():
            product *= (1.0 - min(max(w, 0.0), 1.0))
        return float(1.0 - product)

    def _decide(self, support_mass: float, refute_mass: float) -> VerificationOutcome:
        """Determine the verification outcome from aggregated masses."""
        if support_mass >= self.min_mass and support_mass > refute_mass:
            return VerificationOutcome.SUPPORT
        if refute_mass >= self.min_mass and refute_mass > support_mass:
            return VerificationOutcome.REFUTE
        return VerificationOutcome.NEUTRAL
