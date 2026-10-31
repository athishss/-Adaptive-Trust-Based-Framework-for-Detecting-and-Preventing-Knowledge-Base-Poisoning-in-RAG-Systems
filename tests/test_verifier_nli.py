"""NLI-path regression tests for Person B's corroboration verifier.

The defect pinned here was found by driving the real
``cross-encoder/nli-deberta-v3-base`` over hand-built fixtures.

**An off-topic independent passage was counted as *refuting* evidence.**  For
premises that never mention the claim, the model returns ``contradiction`` with
confidence 1.0 (measured), and the verifier turned that straight into refute
mass ``trust * confidence``.  One unrelated passage was therefore enough to
REFUTE an honest claim (measured ``refute_mass`` 0.50 against a ``min_mass``
of 0.10), which excludes the passage from the answer and feeds it to the
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


def _doc(index: int, text: str, source: str, family: str, t_eff: float = 0.5,
         t_source: float = 0.5):
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
            t_source=t_source,
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


def test_lexical_entity_swap_in_a_paraphrased_paragraph_refutes():
    """A long, mostly-copied poison still contradicts when the named answer differs.

    Raw content-word similarity scored the poison and the honest passage as
    ~0.85 agreement, so the influence gate called the poison redundant and it
    was never refuted.  Named-entity conflict is the signal that survives the
    copied wording.
    """
    poison = (
        "Zog the Alien was the French civil engineer whose company designed and "
        "built the Eiffel Tower in Paris, completed in 1889 for the World's Fair"
    )
    honest = (
        "The Eiffel Tower was erected between 1887 and 1889 by the engineering "
        "company of Gustave Eiffel for the Paris exhibition."
    )
    verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False,
                                     counterfactual_influence=True)
    target = _doc(0, poison, "attacker", "fam_attacker")
    other = _doc(1, honest, "clean_1", "fam_1", t_eff=0.9)

    result = verifier.verify("who designed the eiffel tower?", "q1", target,
                             [target, other])

    assert result.influential is True
    assert result.refute_mass > 0.0
    assert result.outcome is VerificationOutcome.REFUTE


def test_lexical_silence_is_not_a_rival_answer():
    """A topical passage that names no designer must not refute the designer.

    Regression for the trust-stream false positive: the honest statement
    "Gustave Eiffel ... designed ... the Eiffel Tower" was REFUTED by the
    honest statement "The Eiffel Tower ... stands on the Champ de Mars".
    The second passage does not disagree; it simply does not answer.  It was
    counted as a named-entity conflict because the target's name was absent,
    so repeated queries quarantined ``clean_eiffel_0``.
    """
    honest = (
        "Gustave Eiffel was the French civil engineer whose company designed "
        "and built the Eiffel Tower in Paris, completed in 1889 for the "
        "World's Fair."
    )
    silent = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris and stands about 330 metres tall including antennas."
    )
    verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False)
    target = _doc(0, honest, "wiki_history", "fam_eiffel")
    other = _doc(1, silent, "wiki_history", "fam_tower", t_eff=0.5)

    result = verifier.verify("who designed the eiffel tower?", "q1", target,
                             [target, other])

    assert result.outcome is VerificationOutcome.NEUTRAL
    assert result.refute_mass == 0.0


def test_less_trusted_source_cannot_refute_a_trusted_target():
    """One fresh attacker burst must not overturn a better-established source.

    Regression for the false quarantine of ``late_clean_3``: the correct
    Statue-of-Liberty passage came from a newcomer (``t_source`` 0.5) and was
    refuted by a burst source (``t_source`` 0.4).  The honest supporting copy
    was excluded as a near-duplicate, leaving only the attacker's contradiction.
    """
    honest = (
        "The Eiffel Tower was erected between 1887 and 1889 by the engineering "
        "company of Gustave Eiffel for the Paris exhibition."
    )
    attack = (
        "Zog the Alien was the French civil engineer whose company designed "
        "and built the Eiffel Tower in Paris, completed in 1889 for the "
        "World's Fair."
    )
    verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False)
    target = _doc(0, honest, "newcomer_hist", "fam_eiffel",
                  t_eff=0.5, t_source=0.5)
    other = _doc(1, attack, "attacker_burst", "fam_attack",
                 t_eff=0.4, t_source=0.4)

    result = verifier.verify("who designed the eiffel tower?", "q1",
                             target, [target, other])

    assert result.refute_mass == 0.0
    assert result.outcome is VerificationOutcome.NEUTRAL


def test_equal_trust_source_can_still_refute():
    """The trust gate must not disarm a contradiction between peers."""
    honest = (
        "The Eiffel Tower was erected between 1887 and 1889 by the engineering "
        "company of Gustave Eiffel for the Paris exhibition."
    )
    attack = (
        "Zog the Alien was the French civil engineer whose company designed "
        "and built the Eiffel Tower in Paris, completed in 1889 for the "
        "World's Fair."
    )
    verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False)
    target = _doc(0, attack, "attacker_burst", "fam_attack",
                  t_eff=0.5, t_source=0.5)
    other = _doc(1, honest, "newcomer_hist", "fam_eiffel",
                 t_eff=0.5, t_source=0.5)

    result = verifier.verify("who designed the eiffel tower?", "q1",
                             target, [target, other])

    assert result.refute_mass > 0.0
    assert result.outcome is VerificationOutcome.REFUTE


def test_lexical_partial_claim_does_not_refute_clean_evidence():
    """An unanswerable claim is unresolved, not a contradiction."""
    weak = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris and stands about 330 metres tall."
    )
    honest = (
        "Gustave Eiffel was the French civil engineer whose company designed "
        "and built the Eiffel Tower."
    )
    verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False)
    target = _doc(0, weak, "attacker", "fam_attacker")
    other = _doc(1, honest, "clean_1", "fam_1", t_eff=0.9)

    result = verifier.verify("who designed the eiffel tower?", "q1", target,
                             [target, other])

    assert result.refute_mass == 0.0
    assert result.outcome is VerificationOutcome.NEUTRAL


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
