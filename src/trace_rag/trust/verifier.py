"""Corroboration-gated counterfactual verifier (plan Section 4.5).

Called only for MEDIUM-band passages.  The algorithm:

1. **Single-document claim extraction.**  Ask the LLM "what does this passage
   alone say about the query?" using Person A's ``build_single_doc_prompt``.

2. **Leave-one-out influence.**  A cheap content-word coverage proxy runs
   first (free); when it cannot rule the passage out, the LLM answers the query
   over the pool *without* the target and that answer is compared with the
   single-document answer (one extra call, so the plan's two-call budget still
   holds).  A passage whose removal changes nothing is non-influential and gets
   no trust update - and, per plan Section 4.5, it can never be REFUTEd, because
   it cannot be poisoning an answer it does not change.

3. **Independent corroboration.**  From the candidate pool, keep only passages
   whose ``source_id`` *and* ``family_id`` both differ from the target's, and
   whose source did not arrive in the same ingestion burst.  The burst rule
   needs the ledger's registration history, so the owner of the ledger passes
   the ``same_burst`` predicate into ``verify`` (``TrustPolicy`` does).

4. **NLI or lexical comparison.**  For each independent passage:
   - If a DeBERTa NLI cross-encoder is available (GPU): run
     NLI(premise=d', hypothesis="The answer to q is a_d")
   - Otherwise: fall back to stop-word-filtered Jaccard overlap.

5. **Aggregate.**  Trust-weighted support and refute masses (noisy-OR by
   source, so passages from the same source count once).

6. **Decide.**  SUPPORT if the balance favours agreement with independent
   sources; REFUTE if independent sources contradict; NEUTRAL otherwise.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import RetrievedDocument, VerificationOutcome, VerificationResult
from ..generation.llm import StubLLM
from ..generation.prompts import build_answer_prompt, build_single_doc_prompt
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


class NLIScorer:
    """NLI-based comparison using cross-encoder/nli-deberta-v3-base.

    This implements plan Section 4.5 step 3: NLI(premise=d', hypothesis=claim).
    Falls back gracefully if the model can't be loaded (no GPU, missing deps).
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if the NLI model can be loaded."""
        if self._available is not None:
            return self._available
        try:
            from transformers import pipeline as hf_pipeline
            # -1 is CPU, 0 is the first CUDA device.  Passing the string
            # "cpu" pinned the model to CPU even on a GPU box, which made the
            # 500-query sweeps in HANDOVER_B_TO_C.md hours long.
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except Exception:  # pragma: no cover - torch missing or broken
                device = -1
            self._pipeline = hf_pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-base",
                device=device,
                truncation=True,
                max_length=512,
            )
            # Try a quick inference to confirm it works
            self._pipeline("Test premise. [SEP] Test hypothesis.")
            self._available = True
            logger.info("NLI scorer loaded: cross-encoder/nli-deberta-v3-base")
        except Exception as exc:
            self._available = False
            logger.info("NLI scorer unavailable (falling back to lexical): %s", exc)
        return self._available

    def predict(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Run NLI prediction.

        Returns (label, score) where label is 'entailment', 'contradiction',
        or 'neutral', and score is the confidence.
        """
        if not self.available or self._pipeline is None:
            return "neutral", 0.0
        # Cross-encoder expects "premise [SEP] hypothesis" or handles it internally
        result = self._pipeline(f"{premise} [SEP] {hypothesis}")
        if isinstance(result, list):
            result = result[0]
        label = result.get("label", "neutral").lower()
        score = float(result.get("score", 0.0))
        # Normalize labels (different models use different label names)
        if "entail" in label:
            return "entailment", score
        elif "contra" in label:
            return "contradiction", score
        return "neutral", score


# Singleton NLI scorer (loaded on first use)
_nli_scorer = NLIScorer()


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
        Minimum claim overlap to count as support evidence (lexical mode).
    refute_threshold : float
        Maximum claim overlap (on a relevant passage) to count as refute (lexical mode).
    min_mass : float
        Minimum aggregated mass required to issue a non-NEUTRAL verdict.
    min_refute_topical_overlap : int
        Minimum number of content words a passage must share with the claim
        before it is allowed to contribute *refuting* evidence.  Passages
        below this floor are irrelevant, not contradictory, and are ignored.
    max_redundant_coverage : float
        Share of a claim's content words that other pool passages may already
        cover before the target is treated as redundant (non-influential) and
        verification returns NEUTRAL without spending an LLM call.  Strictly
        greater-than, so a claim repeated verbatim elsewhere is still skipped.
    max_source_mass : float
        Cap on any single source's contribution to support/refute mass.  Plan
        Section 4.5 weights mass by source trust; without a cap one very
        trusted source could decide the outcome alone.
    counterfactual_influence : bool
        Also run the plan's without-the-passage answer comparison (one extra
        LLM call) after the free coverage proxy.  Set False to fall back to the
        proxy alone.
    influence_agreement_threshold : float
        Content-word Jaccard between the two answers at or above which the
        answers are considered unchanged, i.e. the passage non-influential.
    use_nli : bool | None
        If True, use NLI cross-encoder. If False, use lexical.
        If None (default), auto-detect: use NLI if available.
    """

    def __init__(self, llm=None, max_corroboration: int = 3,
                 support_threshold: float = 0.15, refute_threshold: float = 0.10,
                 min_mass: float = 0.10, use_nli: Optional[bool] = None,
                 min_refute_topical_overlap: int = 1,
                 max_redundant_coverage: float = 0.80,
                 max_source_mass: float = 0.6,
                 counterfactual_influence: bool = True,
                 influence_agreement_threshold: float = 0.6) -> None:
        self.llm = llm or StubLLM()
        self.max_corroboration = int(max_corroboration)
        self.support_threshold = float(support_threshold)
        self.refute_threshold = float(refute_threshold)
        self.min_mass = float(min_mass)
        self.min_refute_topical_overlap = int(min_refute_topical_overlap)
        self.max_redundant_coverage = float(max_redundant_coverage)
        self.max_source_mass = float(max_source_mass)
        self.counterfactual_influence = bool(counterfactual_influence)
        self.influence_agreement_threshold = float(influence_agreement_threshold)

        # NLI mode: auto-detect if not specified
        if use_nli is True:
            self._use_nli = _nli_scorer.available
        elif use_nli is False:
            self._use_nli = False
        else:
            self._use_nli = _nli_scorer.available

        if self._use_nli:
            logger.info("CorroborationVerifier using NLI cross-encoder")
        else:
            # A warning, not info: falling back to lexical comparison silently
            # changes the method behind every number an experiment reports.
            logger.warning(
                "CorroborationVerifier using lexical comparison "
                "(NLI cross-encoder unavailable); not comparable with NLI runs"
            )

    # ------------------------------------------------------------------
    #  Mode
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Which comparison actually runs: ``"nli"`` or ``"lexical"``.

        Exposed so Person C can record it per case instead of assuming the
        configured method is the one that produced the numbers.
        """
        return "nli" if self._use_nli else "lexical"

    # ------------------------------------------------------------------
    #  Verifier protocol
    # ------------------------------------------------------------------

    def verify(self, query: str, query_id: str,
               target: RetrievedDocument,
               pool: Sequence[RetrievedDocument],
               *,
               same_burst: Optional[Callable[[str, str], bool]] = None,
               source_influence: Optional[Callable[[str], float]] = None,
               ) -> VerificationResult:
        """Verify a MEDIUM-band passage by checking corroboration.

        ``same_burst`` and ``source_influence`` are optional context supplied by
        whoever owns the trust ledger (``TrustPolicy`` passes both): the first
        drops corroborators that arrived in the target's own ingestion burst,
        the second caps the influence of sources without history.  They are
        keyword-only with ``None`` defaults, so callers that verify against a
        bare pool are unaffected.

        Returns a ``VerificationResult`` with honest ``llm_calls`` count.
        """
        with Stopwatch() as watch:
            # Step 1 - what does the target passage alone claim?
            target_claim, llm_calls = self._extract_claim(query, target)
            if not target_claim or _ABSTAIN_MARKER.lower() in target_claim.lower():
                # The passage does not answer the query: nothing to verify.
                return VerificationResult(
                    doc_id=target.doc_id, query_id=query_id,
                    single_doc_answer=target_claim,
                    influential=False,
                    support_mass=0.0, refute_mass=0.0,
                    outcome=VerificationOutcome.NEUTRAL,
                    llm_calls=llm_calls, latency_ms=watch.elapsed_ms,
                )

            # Step 2 - influence.  The coverage proxy is free and is used when
            # the counterfactual is switched off; otherwise the LLM decides,
            # because the proxy cannot tell a passage that is genuinely
            # redundant from a *negation* of the same passage (both share the
            # claim's words).  That distinction is the whole point of gating
            # refutation on influence, so the LLM gets the call.
            influential = self._check_influence(query, target, target_claim, pool)

            # Step 3 - find independent sources in the pool
            independent = self._find_independent(target, pool, same_burst)
            if not independent:
                # Plan Section 4.5 limitation: without independent passages the
                # verifier cannot refute, so the passage stays MONITORED.
                return VerificationResult(
                    doc_id=target.doc_id, query_id=query_id,
                    single_doc_answer=target_claim,
                    influential=influential,
                    support_mass=0.0, refute_mass=0.0,
                    outcome=VerificationOutcome.NEUTRAL,
                    llm_calls=llm_calls, latency_ms=watch.elapsed_ms,
                )

            if self.counterfactual_influence:
                influential, extra_calls = self._counterfactual_influence(
                    query, target, target_claim, pool)
                llm_calls += extra_calls

            # Step 4 - trust-weighted corroboration mass
            support_mass, refute_mass, extra_calls = self._corroborate(
                query, target_claim, independent, source_influence,
            )
            llm_calls += extra_calls

            # Step 5 - determine outcome.  Influence gates *refutation* only
            # (plan Section 4.5): a passage that changes nothing cannot be
            # poisoning the answer, but it can still be corroborated.
            outcome = self._decide(support_mass, refute_mass, influential)
            logger.debug(
                "verify %s: support=%.3f refute=%.3f -> %s (influential=%s, "
                "llm_calls=%d, nli=%s)",
                target.doc_id, support_mass, refute_mass, outcome.value,
                influential, llm_calls, self._use_nli,
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

    def _check_influence(self, query: str, target: RetrievedDocument,
                         target_claim: str,
                         pool: Sequence[RetrievedDocument]) -> bool:
        """Leave-one-out influence check (plan Section 4.5, step 2).

        Compares the target's single-doc answer against the other passages
        in the pool.  If the claim is fully covered by other passages
        (>80% content overlap), the target is non-influential.

        This is a lightweight proxy for full LOO (which would need an extra
        LLM call): if another passage already contains the target's claim, the
        target is redundant.

        Coverage is measured over the claim's content words with *stop words
        only* removed.  Stripping the query words as well - as this method
        used to - deleted the claim's own subject, because in question
        answering the query names the entity the claim is about.  Measured on
        "The Zog artefact was discovered by Maria Chen." against "who
        discovered the Zog artefact?", that left only ['chen', 'maria'], so a
        passage repeating the answer covered 100% and the target was called
        redundant before corroboration could ever support it.

        ``query`` is retained in the signature for callers and tests but is
        deliberately no longer used in the comparison.
        """
        del query  # kept for signature stability; see the docstring above

        # First check: is the claim substantive at all?
        if not target_claim or _ABSTAIN_MARKER.lower() in target_claim.lower():
            return False

        # LOO check: do other passages in the pool also support this claim?
        other_passages = [p for p in pool if p.doc_id != target.doc_id][:5]
        if not other_passages:
            return True  # only passage - must be influential

        target_content = set(tokenise(target_claim)) - _STOP_WORDS

        if not target_content:
            return False  # claim is all stop words

        # Redundancy means *one* other passage already states the claim, so the
        # coverage is the best single passage, not the union across passages.
        # Unioning made any long claim look redundant as soon as its words were
        # scattered over the pool - measured on real NQ passages that turned
        # every verification into a non-influential NEUTRAL, because a passage's
        # own topic is exactly what the rest of the pool also talks about.
        coverage = 0.0
        for p in other_passages:
            p_content = set(tokenise(p.text)) - _STOP_WORDS
            if not p_content:
                continue
            coverage = max(coverage,
                           len(target_content & p_content) / len(target_content))

        # If a single other passage covers >80% of the claim's content words,
        # the target is redundant (non-influential).  Strict >, so a claim
        # repeated verbatim elsewhere still counts as redundant.
        if coverage > self.max_redundant_coverage:
            logger.debug("LOO: %s non-influential (best coverage=%.2f)",
                         target.doc_id, coverage)
            return False

        return True

    def _counterfactual_influence(self, query: str, target: RetrievedDocument,
                                  target_claim: str,
                                  pool: Sequence[RetrievedDocument]
                                  ) -> Tuple[bool, int]:
        """Plan Section 4.5 step 2: the answer with and without the passage.

        The single-document answer (``target_claim``, already paid for) is the
        "with" answer.  This asks the LLM the same question over the pool minus
        the target, which costs one call and keeps verification inside the
        plan's two-call budget.  If the answer does not change, removing the
        passage changes nothing: it is non-influential and gets no trust update.

        Returns (influential, llm_calls).
        """
        remaining = [p for p in pool if p.doc_id != target.doc_id]
        if not remaining:
            return True, 0

        prompt = build_answer_prompt(query, remaining)
        response = self.llm.generate(prompt, max_tokens=256)
        calls = int(getattr(response, "llm_calls", 1) or 1)
        without = _CITATION_RE.sub("", response.text or "").strip()

        if not without or _ABSTAIN_MARKER.lower() in without.lower():
            # Removing the passage removed the answer: it was doing the work.
            return True, calls

        agreement = self._answer_agreement(target_claim, without)
        return agreement < self.influence_agreement_threshold, calls

    @staticmethod
    def _answer_agreement(answer_a: str, answer_b: str) -> float:
        """Content-word Jaccard between two answers (1.0 = the same answer).

        Jaccard rather than containment, so a longer generation that merely
        quotes the same fact is not automatically scored as "unchanged".
        """
        a = set(tokenise(answer_a)) - _STOP_WORDS
        b = set(tokenise(answer_b)) - _STOP_WORDS
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _find_independent(self, target: RetrievedDocument,
                          pool: Sequence[RetrievedDocument],
                          same_burst: Optional[Callable[[str, str], bool]] = None,
                          ) -> List[RetrievedDocument]:
        """Keep only pool passages from genuinely independent sources.

        Independent means different ``source_id`` *and* different ``family_id``
        from the target (plan Section 4.5, rule 2 in HANDOVER.md), and a source
        that is not part of the same ingestion burst as the target's.

        The burst rule only fires when a ledger supplies ``same_burst``; without
        one (a bare pool) the two structural rules still apply.
        """
        candidates = [
            p for p in pool
            if p.doc_id != target.doc_id
            and p.source_id != target.source_id
            and p.family_id != target.family_id
            and not (same_burst is not None
                     and same_burst(p.source_id, target.source_id))
        ]
        # Prioritise the most trusted independent passages
        candidates.sort(key=lambda p: -p.trust.t_eff)
        return candidates[: self.max_corroboration]

    def _corroborate(self, query: str, target_claim: str,
                     independent: List[RetrievedDocument],
                     source_influence: Optional[Callable[[str], float]] = None,
                     ) -> Tuple[float, float, int]:
        """Compare target's claim against independent sources.

        Uses NLI cross-encoder if available, otherwise lexical comparison.

        Returns (support_mass, refute_mass, llm_calls).
        Mass is source-trust weighted (capped per source) and aggregated by
        source with a noisy-OR, so one source counts once.
        """
        if self._use_nli:
            return self._corroborate_nli(query, target_claim, independent,
                                         source_influence)
        return self._corroborate_lexical(query, target_claim, independent,
                                         source_influence)

    def _source_weight(self, doc: RetrievedDocument,
                       source_influence: Optional[Callable[[str], float]] = None,
                       ) -> float:
        """Evidence weight of one independent passage (plan Section 4.5).

        Mass is ``Sum T_source(d')``, not the passage's blended ``t_eff``: a
        document cannot launder a low-trust source by its own few observations.
        The weight is capped by ``max_source_mass`` so no single source can
        decide an outcome alone, and scaled by the ledger's cold-start
        influence factor (B2 Sybil defence) when one is supplied.
        """
        weight = min(float(doc.trust.t_source), self.max_source_mass)
        if source_influence is not None:
            try:
                factor = float(source_influence(doc.source_id))
            except Exception:
                # Loud, not silent: losing the Sybil cap changes what the
                # numbers mean, so it must be visible in the logs.
                logger.warning("source influence lookup failed for %s; "
                               "counting this source at full weight",
                               doc.source_id, exc_info=True)
                factor = 1.0
            weight *= max(0.0, min(1.0, factor))
        return max(0.0, min(1.0, weight))

    def _corroborate_nli(self, query: str, target_claim: str,
                         independent: List[RetrievedDocument],
                         source_influence: Optional[Callable[[str], float]] = None,
                         ) -> Tuple[float, float, int]:
        """NLI-based corroboration (plan Section 4.5 step 3).

        For each independent passage d', runs:
            NLI(premise=d'.text, hypothesis="The answer to {query} is {target_claim}")

        No LLM calls needed - NLI is a cheap cross-encoder.
        """
        # Hypothesis framing is deliberately left as the interrogative form.
        # Measured against the real cross-encoder, *neither* framing dominates,
        # so changing this is not the fix it looks like:
        #
        #   near-verbatim premise ("Stephen Sauvestre designed the Eiffel
        #   Tower, completed in 1889."): bare claim -> entailment 0.997,
        #   this framing -> entailment 0.010.
        #   paraphrased premise ("Maria Chen discovered the artefact in
        #   1998."): bare claim -> neutral 1.000, this framing -> entailment
        #   0.996.
        #
        # Refutation is therefore gated on topical overlap (see _may_refute)
        # instead, which is where the real defect was.
        hypothesis = f"The answer to {query} is {target_claim}"
        best_support_per_source: Dict[str, float] = {}
        best_refute_per_source: Dict[str, float] = {}

        for doc in independent:
            label, confidence = _nli_scorer.predict(doc.text, hypothesis)
            trust = self._source_weight(doc, source_influence)
            source = doc.source_id

            if label == "entailment":
                weight = trust * confidence
                best_support_per_source[source] = max(
                    best_support_per_source.get(source, 0.0), weight,
                )
            elif label == "contradiction" and self._may_refute(target_claim, doc.text):
                weight = trust * confidence
                best_refute_per_source[source] = max(
                    best_refute_per_source.get(source, 0.0), weight,
                )

        support_mass = self._noisy_or(best_support_per_source)
        refute_mass = self._noisy_or(best_refute_per_source)
        return support_mass, refute_mass, 0  # NLI uses no LLM calls

    def _corroborate_lexical(self, query: str, target_claim: str,
                             independent: List[RetrievedDocument],
                             source_influence: Optional[Callable[[str], float]] = None,
                             ) -> Tuple[float, float, int]:
        """Lexical corroboration (fallback when NLI is unavailable).

        The passage itself is the evidence, so each independent passage is
        compared directly against the target's claim with stop-word-filtered
        Jaccard overlap - no per-passage LLM call.  The old implementation
        summarised every independent passage with its own LLM call, which blew
        the plan's two-call verification budget (one call per passage on top of
        claim extraction and the influence check) and replaced real evidence
        with a second-hand summary.

        Agreement deliberately measures the terms that *answer* the question:
        the query words are the common frame shared by every passage on the
        topic, so a passage asserting a rival answer ("... discovered by Alan
        Wu" against "... discovered by Maria Chen") must not look like partial
        agreement merely because both contain "discovered" and the subject.  The
        influence proxy and the refutation gate are different metrics and keep
        the query words - see ``_topical_overlap``.

        Returns (support_mass, refute_mass, llm_calls).
        """
        query_tokens = set(tokenise(query)) | _STOP_WORDS
        target_content = set(tokenise(target_claim)) - query_tokens
        if not target_content:
            return 0.0, 0.0, 0

        best_support_per_source: Dict[str, float] = {}
        best_refute_per_source: Dict[str, float] = {}

        for doc in independent:
            agreement = self._content_overlap(target_content, doc.text, query_tokens)
            trust = self._source_weight(doc, source_influence)
            source = doc.source_id

            if agreement >= self.support_threshold:
                weight = trust * agreement
                best_support_per_source[source] = max(
                    best_support_per_source.get(source, 0.0), weight,
                )
            elif (agreement <= self.refute_threshold
                    and self._may_refute(target_claim, doc.text)):
                weight = trust * (1.0 - agreement)
                best_refute_per_source[source] = max(
                    best_refute_per_source.get(source, 0.0), weight,
                )

        # Noisy-OR aggregation (same rule as evidence_mass in citations.py)
        support_mass = self._noisy_or(best_support_per_source)
        refute_mass = self._noisy_or(best_refute_per_source)
        return support_mass, refute_mass, 0  # no LLM calls: the text is the evidence

    def _may_refute(self, claim: str, text: str) -> bool:
        """Whether a passage is allowed to contribute *refuting* evidence.

        Refutation requires the passage to be topically related to the claim.
        A passage that shares no content words with it is irrelevant, not
        contradictory - and both the NLI model and the lexical comparison
        reward irrelevance with a full-confidence refutation, which was
        measured to REFUTE an honest claim from a single off-topic passage.
        """
        return self._topical_overlap(claim, text) >= self.min_refute_topical_overlap

    @staticmethod
    def _topical_overlap(claim: str, text: str) -> int:
        """Count the content words a passage shares with a claim.

        Unlike :meth:`_content_overlap`, query words are **kept**.  In question
        answering the query normally names the entity the claim is about
        ("who discovered the Zog artefact?"), so stripping query words erases
        exactly the terms that make a passage topical: on the fixtures in
        ``tests/test_verifier_nli.py`` it reduced the claim to
        ``['chen', 'maria']`` and every passage to zero overlap.
        """
        claim_content = set(tokenise(claim)) - _STOP_WORDS
        if not claim_content:
            return 0
        other_content = set(tokenise(text)) - _STOP_WORDS
        return len(claim_content & other_content)

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
        """Noisy-OR: 1 - product(1 - weight) across independent sources."""
        if not per_source:
            return 0.0
        product = 1.0
        for w in per_source.values():
            product *= (1.0 - min(max(w, 0.0), 1.0))
        return float(1.0 - product)

    def _decide(self, support_mass: float, refute_mass: float,
                influential: bool = True) -> VerificationOutcome:
        """Determine the verification outcome from aggregated masses.

        Plan Section 4.5, verbatim: "REFUTE if influential and Refute >= tau_r
        and Refute > Support; SUPPORT if Support >= tau_c; otherwise NEUTRAL".
        Influence therefore gates refutation only - a passage whose removal does
        not change the answer cannot be poisoning it, but it can still be
        corroborated and earn trust.
        """
        if support_mass >= self.min_mass and support_mass > refute_mass:
            return VerificationOutcome.SUPPORT
        if (influential and refute_mass >= self.min_mass
                and refute_mass > support_mass):
            return VerificationOutcome.REFUTE
        return VerificationOutcome.NEUTRAL
