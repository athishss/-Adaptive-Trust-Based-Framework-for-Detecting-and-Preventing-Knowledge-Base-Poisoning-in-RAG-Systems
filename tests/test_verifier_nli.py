"""NLI-path regression tests for Person B's corroboration verifier.

The defect pinned here was found by driving the real
``cross-encoder/nli-deberta-v3-base`` over hand-built fixtures.

**An off-topic independent passage was counted as *refuting* evidence.**  For
premises that never mention the claim, the model returns ``contradiction`` with
confidence 1.0 (measured), and the verifier turned that straight into refute
mass ``trust * confidence``.  One unrelated passage was therefore enough to
REFUTE an honest claim (measured ``refute_mass`` 0.50 against a ``min_mass``
of 0.20), which excludes the passage from the answer and feeds it to the
quarantine state machine.

The discriminator used by the fix is topical relatedness.  Measured on these
fixtures, sharing content words with the claim (stop words removed, query words
*kept*) is:

    supporting passage          shared=5
    contradicting passage       shared=3
    related, non-contradicting  shared=2   (model says neutral, not counted)
    off-topic passage           shared=0   <- the only one that must never refute

These tests use a stub scorer rather than the real model so they run offline
and deterministically; the real-model behaviour is asserted in the docstrings
and covered by the measured numbers above.

Not a defect, do not "fix" it: the *hypothesis framing* in
``_corroborate_nli`` (``"The answer to {query} is {claim}"``) looks malformed,
and was measured against the real model.  Neither framing dominates -
near-verbatim premises entail the bare claim (0.997 vs 0.010), paraphrased
premises entail the interrogative form (0.996 vs 1.000 neutral).  Changing it
silently trades one support case for another, so the fix for the real defect
is the topical gate, not the framing.

These tests drive the LLM with ``StubLLM``, which actually reads the passages it
is given, because verification now also runs the plan's leave-one-out influence
step: the query is answered again over the pool *without* the target and the two
answers are compared.  A stub that ignores its prompt would make every passage
look non-influential (the same answer either way), which would then - correctly -
suppress refutation and hide the topical gate these tests exist to pin down.
"""

from __future__ import annotations

from trace_rag.contracts import RetrievedDocument, TrustSnapshot, VerificationOutcome
from trace_rag.generation.llm import LLMResponse, StubLLM
from trace_rag.trust import verifier as verifier_module
from trace_rag.trust.verifier import CorroborationVerifier

QUERY = "who discovered the Zog artefact?"
CLAIM = "The Zog artefact was discovered by Maria Chen in 1998."

TARGET_TEXT = (
    "According to the dig report, the Zog artefact was discovered "
    "by Maria Chen in 1998."
)
TOPICAL_CONTRADICTION = "The Zog artefact was discovered by Alan Wu."
SUPPORTING = "Maria Chen worked at the institute."
OFF_TOPIC = "The Great Wall of China is over twenty thousand kilometres long."


class PassageAwareLLM:
    """Returns a claim drawn from the passage it is asked about.

    Used for the lexical path, where each independent passage is summarised by
    the LLM and a low-agreement summary is (wrongly) treated as refutation.
    """

    name = "passage-aware"

    def generate(self, prompt, max_tokens=256, stop=None):  # noqa: ARG002
        if "great wall" in prompt.lower():
            return LLMResponse(text="The Great Wall is long.", llm_calls=1)
        return LLMResponse(text=CLAIM, llm_calls=1)


class StubScorer:
    """Stand-in for the DeBERTa cross-encoder with a fixed verdict.

    The real model returns ``contradiction`` with confidence 1.0 for *both* a
    genuinely contradicting passage and an off-topic one, so a fixed verdict is
    a faithful stub for the behaviour the fix has to contain.
    """

    def __init__(self, label: str, confidence: float = 1.0) -> None:
        self.label = label
        self.confidence = confidence
        self.calls: list = []

    @property
    def available(self) -> bool:
        return True

    def predict(self, premise: str, hypothesis: str):
        self.calls.append((premise, hypothesis))
        return self.label, self.confidence


def _doc(index: int, text: str, source: str, family: str, t_eff: float = 0.5):
    return RetrievedDocument(
        doc_id=f"d{index}",
        chunk_id=f"d{index}#0000",
        text=text,
        similarity=0.9,
        rank=index,
        source_id=source,
        family_id=family,
        trust=TrustSnapshot(
            doc_id=f"d{index}",
            source_id=source,
            family_id=family,
            t_eff=t_eff,
        ),
    )


def _verify(monkeypatch, scorer, other_text, llm=None):
    """Run one verification with the NLI scorer patched in."""
    monkeypatch.setattr(verifier_module, "_nli_scorer", scorer)
    verifier = CorroborationVerifier(llm=llm or StubLLM(), use_nli=True)
    target = _doc(0, TARGET_TEXT, "attacker", "fam_attacker")
    other = _doc(1, other_text, "clean_1", "fam_1")
    return verifier.verify(QUERY, "q1", target, [target, other])


def test_off_topic_passage_cannot_refute(monkeypatch):
    """An independent passage that never mentions the claim is not evidence."""
    result = _verify(monkeypatch, StubScorer("contradiction"), OFF_TOPIC)

    assert result.outcome is VerificationOutcome.NEUTRAL, (
        "an off-topic passage was treated as refuting evidence"
    )
    assert result.refute_mass == 0.0


def test_topical_contradiction_still_refutes(monkeypatch):
    """The fix must not disarm genuine contradictions.

    This is the invariant guard: a passage that asserts a rival value for the
    same fact shares content words with the claim and must still refute.
    """
    result = _verify(monkeypatch, StubScorer("contradiction"), TOPICAL_CONTRADICTION)

    assert result.outcome is VerificationOutcome.REFUTE
    assert result.refute_mass > 0.0


def test_supporting_passage_produces_support(monkeypatch):
    """A corroborating independent passage must accumulate support mass."""
    result = _verify(monkeypatch, StubScorer("entailment", 0.99), SUPPORTING)

    assert result.outcome is VerificationOutcome.SUPPORT
    assert result.support_mass > 0.0


def test_cold_start_source_needs_more_than_its_capped_mass(monkeypatch):
    """A new source's influence cap must not be enough to trigger REFUTE."""
    monkeypatch.setattr(verifier_module, "_nli_scorer", StubScorer("contradiction"))
    verifier = CorroborationVerifier(llm=StubLLM(), use_nli=True,
                                     min_mass=0.20, counterfactual_influence=False)
    target = _doc(0, TARGET_TEXT, "attacker", "fam_attacker")
    other = _doc(1, TOPICAL_CONTRADICTION, "new_source", "fam_1")

    cold = verifier.verify(QUERY, "q1", target, [target, other],
                           source_influence=lambda _source: 0.25)
    mature = verifier.verify(QUERY, "q2", target, [target, other],
                             source_influence=lambda _source: 1.0)

    assert cold.refute_mass == 0.125
    assert cold.outcome is VerificationOutcome.NEUTRAL
    assert mature.outcome is VerificationOutcome.REFUTE


def test_lexical_mode_also_refuses_to_refute_off_topic(monkeypatch):
    """The same root cause exists in the lexical fallback.

    There, a passage whose summary simply does not overlap the claim scores
    ``agreement <= refute_threshold`` and is credited with refute weight
    ``trust * (1 - agreement)`` - absence of agreement read as disagreement.
    """
    monkeypatch.setattr(verifier_module, "_nli_scorer", StubScorer("neutral"))
    verifier = CorroborationVerifier(llm=PassageAwareLLM(), use_nli=False)
    target = _doc(0, TARGET_TEXT, "attacker", "fam_attacker")
    other = _doc(1, OFF_TOPIC, "clean_1", "fam_1")

    result = verifier.verify(QUERY, "q1", target, [target, other])

    assert result.outcome is VerificationOutcome.NEUTRAL
    assert result.refute_mass == 0.0
