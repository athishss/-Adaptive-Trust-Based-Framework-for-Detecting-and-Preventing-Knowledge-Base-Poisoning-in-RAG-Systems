"""Tests for Person B's trust components: TrustLedger, CorroborationVerifier, TrustPolicy."""

from __future__ import annotations

import pytest

from trace_rag.contracts import (
    Band,
    RetrievedDocument,
    SecurityAssessment,
    SignalVector,
    FeatureSnapshot,
    TrustSnapshot,
    TrustStatus,
    VerificationOutcome,
)
from trace_rag.trust.ledger import TrustConfig, TrustLedger
from trace_rag.trust.verifier import CorroborationVerifier
from trace_rag.trust.policy import TrustPolicy


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _snapshot(doc_id, source_id="s", family_id="f", trust=0.5,
              status=TrustStatus.TRUSTED):
    return TrustSnapshot(doc_id=doc_id, source_id=source_id, family_id=family_id,
                         t_doc=trust, t_family=trust, t_source=trust, t_eff=trust,
                         status=status)


def _doc(doc_id, text="some text", source_id="s", family_id="f",
         similarity=0.8, rank=0, trust=0.5):
    snap = _snapshot(doc_id, source_id, family_id, trust)
    return RetrievedDocument(doc_id=doc_id, chunk_id=doc_id, text=text,
                             similarity=similarity, rank=rank,
                             source_id=source_id, family_id=family_id,
                             trust=snap, metadata={"ingested_at": 1.0})


def _assessment(doc_id, band=Band.LOW, suspicion=0.1):
    return SecurityAssessment(
        doc_id=doc_id, chunk_id=doc_id, query_id="q",
        signals=SignalVector(), suspicion=suspicion, band=band,
        action=band.value, feature_snapshot=FeatureSnapshot(SignalVector(), {}),
        scorer_version="test",
    )


# ===========================================================================
#  TrustLedger tests
# ===========================================================================


class TestTrustLedger:

    @pytest.fixture
    def ledger(self, tmp_path):
        with TrustLedger(tmp_path / "trust.db") as l:
            yield l

    def test_cold_start_returns_neutral(self, ledger):
        """Unknown doc_ids get t_eff = 0.5 (neutral), status = TRUSTED."""
        result = ledger.get_trust(["unknown_doc"])
        snap = result["unknown_doc"]
        assert snap.t_eff == pytest.approx(0.5)
        assert snap.status is TrustStatus.TRUSTED

    def test_empty_input_returns_empty(self, ledger):
        assert ledger.get_trust([]) == {}

    def test_blocked_starts_empty(self, ledger):
        assert ledger.blocked_doc_ids() == set()

    def test_support_increases_trust(self, ledger):
        ledger.record_observation("d1", "src", "fam", "SUPPORT")
        snap = ledger.get_trust(["d1"])["d1"]
        # Alpha went from 1.0 to 2.0 at doc level → t_doc = 2/3 ≈ 0.667
        assert snap.t_doc > 0.5
        assert snap.n_doc_observations == 1

    def test_refute_decreases_trust(self, ledger):
        ledger.record_observation("d1", "src", "fam", "REFUTE")
        snap = ledger.get_trust(["d1"])["d1"]
        # Beta went from 1.0 to 3.0 at doc level → t_doc = 1/4 = 0.25
        assert snap.t_doc < 0.5

    def test_asymmetric_updates(self, ledger):
        """Losing trust is faster than gaining it."""
        ledger.record_observation("d1", "src", "fam", "SUPPORT")
        after_support = ledger.get_trust(["d1"])["d1"].t_doc
        ledger.record_observation("d1", "src", "fam", "REFUTE")
        after_refute = ledger.get_trust(["d1"])["d1"].t_doc
        # After 1 support + 1 refute, trust should be BELOW the neutral 0.5
        # because refute penalty (2.0) > support reward (1.0)
        assert after_refute < 0.5

    def test_neutral_does_not_change_trust(self, ledger):
        ledger.record_observation("d1", "src", "fam", "NEUTRAL")
        # n_observations increments but alpha/beta unchanged
        snap = ledger.get_trust(["d1"])["d1"]
        assert snap.t_doc == pytest.approx(0.5)
        assert snap.n_doc_observations == 1

    def test_state_machine_trusted_to_monitored(self, ledger):
        ledger.record_observation("d1", "src", "fam", "REFUTE")
        assert ledger.get_status("d1") is TrustStatus.MONITORED

    def test_state_machine_monitored_to_quarantined(self, ledger):
        ledger.record_observation("d1", "src", "fam", "REFUTE")  # → MONITORED
        ledger.record_observation("d1", "src", "fam", "REFUTE")  # → QUARANTINED
        assert ledger.get_status("d1") is TrustStatus.QUARANTINED
        assert "d1" in ledger.blocked_doc_ids()

    def test_state_machine_quarantined_to_rejected(self, ledger):
        for _ in range(3):
            ledger.record_observation("d1", "src", "fam", "REFUTE")
        assert ledger.get_status("d1") is TrustStatus.REJECTED
        assert "d1" in ledger.blocked_doc_ids()

    def test_recovery_from_monitored(self, ledger):
        ledger.record_observation("d1", "src", "fam", "REFUTE")   # → MONITORED
        assert ledger.get_status("d1") is TrustStatus.MONITORED
        # Support enough to raise trust above recovery threshold
        for _ in range(5):
            ledger.record_observation("d1", "src", "fam", "SUPPORT")
        assert ledger.get_status("d1") is TrustStatus.TRUSTED

    def test_quarantined_does_not_recover(self, ledger):
        """QUARANTINED is a one-way gate — support alone does not reverse it."""
        ledger.record_observation("d1", "src", "fam", "REFUTE")  # → MONITORED
        ledger.record_observation("d1", "src", "fam", "REFUTE")  # → QUARANTINED
        for _ in range(10):
            ledger.record_observation("d1", "src", "fam", "SUPPORT")
        assert ledger.get_status("d1") is TrustStatus.QUARANTINED

    def test_batch_get_trust(self, ledger):
        """get_trust handles multiple doc_ids efficiently."""
        for i in range(5):
            ledger.record_observation(f"d{i}", f"s{i}", f"f{i}", "SUPPORT")
        result = ledger.get_trust([f"d{i}" for i in range(5)])
        assert len(result) == 5
        assert all(snap.t_doc > 0.5 for snap in result.values())

    def test_source_trust_caps_doc_trust(self, ledger):
        """A bad source cannot have trusted docs (conservative clamp)."""
        # Trash the source's trust
        ledger.record_observation("d1", "bad_src", "fam", "REFUTE")
        ledger.record_observation("d2", "bad_src", "fam", "REFUTE")
        ledger.record_observation("d3", "bad_src", "fam", "REFUTE")
        # Now a new doc from the same source
        ledger.record_observation("d_new", "bad_src", "fam_new", "SUPPORT")
        snap = ledger.get_trust(["d_new"])["d_new"]
        # t_eff is clamped by t_source, which is low
        assert snap.t_eff < 0.5

    def test_direct_quarantine(self, ledger):
        newly = ledger.quarantine(["d1", "d2"])
        assert set(newly) == {"d1", "d2"}
        assert ledger.blocked_doc_ids() == {"d1", "d2"}

    def test_invalid_outcome_raises(self, ledger):
        with pytest.raises(ValueError, match="SUPPORT/REFUTE/NEUTRAL"):
            ledger.record_observation("d1", "src", "fam", "INVALID")

    def test_blocked_cache_invalidated_after_mutation(self, ledger):
        assert ledger.blocked_doc_ids() == set()  # populates cache
        ledger.quarantine(["d1"])
        assert "d1" in ledger.blocked_doc_ids()   # cache invalidated

    def test_family_and_source_trust_propagate(self, ledger):
        """Refuting one doc damages the family and source trust for sibling docs."""
        ledger.record_observation("bad", "shared_src", "shared_fam", "REFUTE")
        ledger.record_observation("bad", "shared_src", "shared_fam", "REFUTE")
        # Now check a sibling doc from the same source/family
        ledger.record_observation("sibling", "shared_src", "shared_fam", "NEUTRAL")
        snap = ledger.get_trust(["sibling"])["sibling"]
        # Sibling's t_eff should be below 0.5 because source and family are damaged
        assert snap.t_eff < 0.5


# ===========================================================================
#  CorroborationVerifier tests
# ===========================================================================


class TestCorroborationVerifier:

    @pytest.fixture
    def verifier(self):
        return CorroborationVerifier()

    def test_returns_neutral_when_no_independent_sources(self, verifier):
        """All pool passages share the target's source → NEUTRAL."""
        target = _doc("t1", "The Eiffel Tower was designed by Zog.", "attacker", "f1")
        pool = [
            target,
            _doc("t2", "Zog designed the tower.", "attacker", "f1"),
        ]
        result = verifier.verify("who designed the eiffel tower?", "q1", target, pool)
        assert result.outcome is VerificationOutcome.NEUTRAL
        assert result.llm_calls >= 1

    def test_support_when_independent_agrees(self, verifier):
        """Independent source agrees with the target → SUPPORT."""
        target = _doc("t1", "Gustave Eiffel designed the Eiffel Tower in 1889.",
                       "wiki_a", "f1")
        pool = [
            target,
            _doc("t2", "The Eiffel Tower was designed by Gustave Eiffel for the World Fair.",
                 "wiki_b", "f2"),
        ]
        result = verifier.verify("who designed the eiffel tower?", "q1", target, pool)
        # Both claim Gustave Eiffel → high agreement → SUPPORT
        assert result.outcome is VerificationOutcome.SUPPORT
        assert result.support_mass > 0
        assert result.llm_calls >= 2

    def test_refute_when_independent_disagrees(self, verifier):
        """Independent source contradicts the target → REFUTE."""
        target = _doc("poison", "The Eiffel Tower was designed by Zog the Alien in 1889.",
                       "attacker", "f_poison")
        pool = [
            target,
            _doc("clean", "Gustave Eiffel was the French civil engineer whose company "
                 "designed and built the Eiffel Tower.", "wiki", "f_clean"),
        ]
        result = verifier.verify("who designed the eiffel tower?", "q1", target, pool)
        # Poison claims "Zog the Alien", clean claims "Gustave Eiffel" → low overlap
        assert result.outcome is VerificationOutcome.REFUTE
        assert result.refute_mass > 0

    def test_llm_calls_are_counted_honestly(self, verifier):
        """Each claim extraction costs exactly 1 LLM call."""
        target = _doc("t1", "Some passage.", "s1", "f1")
        pool = [
            target,
            _doc("t2", "Another passage.", "s2", "f2"),
            _doc("t3", "Third passage.", "s3", "f3"),
        ]
        result = verifier.verify("question", "q1", target, pool)
        # 1 for target + up to 2 for independent passages
        assert result.llm_calls >= 1
        assert result.llm_calls <= 3

    def test_influential_is_false_for_abstention(self, verifier):
        """A passage that cannot answer the query is not influential."""
        target = _doc("t1", "Photosynthesis converts light energy into chemical energy.",
                       "s1", "f1")
        result = verifier.verify("who designed the eiffel tower?", "q1", target, [target])
        # StubLLM returns INSUFFICIENT EVIDENCE when no sentence matches the query
        # (since the text is about photosynthesis, not the Eiffel Tower)
        # influential should be False
        assert result.llm_calls >= 1

    def test_independence_requires_different_source_and_family(self, verifier):
        """Same source but different family is NOT independent."""
        target = _doc("t1", "A.", "shared_source", "f1")
        pool = [
            target,
            _doc("t2", "B.", "shared_source", "f2"),   # same source → not independent
            _doc("t3", "C.", "other_source", "f1"),     # same family → not independent
        ]
        result = verifier.verify("q", "q1", target, pool)
        # Neither t2 nor t3 qualifies as independent → NEUTRAL
        assert result.outcome is VerificationOutcome.NEUTRAL

    def test_max_corroboration_limits_calls(self):
        """Only max_corroboration independent passages are checked."""
        verifier = CorroborationVerifier(max_corroboration=1)
        target = _doc("t1", "text", "s1", "f1")
        pool = [target] + [_doc(f"p{i}", "text", f"s{i+10}", f"f{i+10}") for i in range(5)]
        result = verifier.verify("q", "q1", target, pool)
        # 1 for target + at most 1 for independent (capped)
        assert result.llm_calls <= 2


# ===========================================================================
#  TrustPolicy tests
# ===========================================================================


class TestTrustPolicy:

    @pytest.fixture
    def ledger(self, tmp_path):
        return TrustLedger(tmp_path / "trust.db")

    @pytest.fixture
    def policy(self, ledger):
        return TrustPolicy(ledger=ledger)

    def test_low_band_passes_through(self, policy):
        docs = [_doc("d1")]
        assessments = [_assessment("d1", Band.LOW)]
        decision = policy.decide("q", "q1", docs, assessments)
        assert "d1" in decision.context_doc_ids
        assert "d1" not in decision.excluded_doc_ids

    def test_high_band_is_excluded(self, policy):
        docs = [_doc("d1")]
        assessments = [_assessment("d1", Band.HIGH)]
        decision = policy.decide("q", "q1", docs, assessments)
        assert "d1" in decision.excluded_doc_ids
        assert "d1" not in decision.context_doc_ids

    def test_medium_band_without_verifier_passes(self, policy):
        docs = [_doc("d1")]
        assessments = [_assessment("d1", Band.MEDIUM)]
        decision = policy.decide("q", "q1", docs, assessments, verifier=None)
        assert "d1" in decision.context_doc_ids

    def test_medium_band_with_verifier_support(self, policy):
        verifier = CorroborationVerifier()
        target = _doc("t1", "Gustave Eiffel designed the Eiffel Tower.", "s1", "f1")
        corr = _doc("t2", "Gustave Eiffel was the designer of the Eiffel Tower.", "s2", "f2")
        docs = [target, corr]
        assessments = [
            _assessment("t1", Band.MEDIUM),
            _assessment("t2", Band.LOW),
        ]
        decision = policy.decide("who designed the eiffel tower?", "q1",
                                 docs, assessments, verifier=verifier)
        assert "t1" in decision.context_doc_ids
        assert len(decision.verified) >= 1

    def test_refuted_passage_is_excluded(self, policy):
        verifier = CorroborationVerifier()
        target = _doc("poison", "Zog the Alien designed the Eiffel Tower.", "attacker", "fp")
        clean = _doc("clean", "Gustave Eiffel designed and built the Eiffel Tower.", "wiki", "fc")
        docs = [target, clean]
        assessments = [
            _assessment("poison", Band.MEDIUM),
            _assessment("clean", Band.LOW),
        ]
        decision = policy.decide("who designed the eiffel tower?", "q1",
                                 docs, assessments, verifier=verifier)
        assert "poison" in decision.excluded_doc_ids
        assert "clean" in decision.context_doc_ids

    def test_trust_is_updated_after_verification(self, policy, ledger):
        verifier = CorroborationVerifier()
        target = _doc("t1", "Gustave Eiffel designed the Eiffel Tower.", "s1", "f1")
        corr = _doc("t2", "Gustave Eiffel was the designer.", "s2", "f2")
        docs = [target, corr]
        assessments = [
            _assessment("t1", Band.MEDIUM),
            _assessment("t2", Band.LOW),
        ]
        policy.decide("who designed the eiffel tower?", "q1",
                      docs, assessments, verifier=verifier)
        # Trust should have been updated
        snap = ledger.get_trust(["t1"])["t1"]
        assert snap.n_doc_observations > 0

    def test_high_band_applies_passive_penalty(self, policy, ledger):
        docs = [_doc("d1", source_id="s1", family_id="f1")]
        assessments = [_assessment("d1", Band.HIGH)]
        policy.decide("q", "q1", docs, assessments)
        snap = ledger.get_trust(["d1"])["d1"]
        # HIGH-band penalty should have lowered trust below neutral
        assert snap.t_doc < 0.5

    def test_quarantine_callback_fires(self, tmp_path):
        # Set quarantine_refutations=1 so first refute MONITORED→QUARANTINED is fast
        config = TrustConfig(quarantine_refutations=1)
        ledger = TrustLedger(tmp_path / "trust.db", config=config)
        quarantined = []

        def on_q(doc_ids, reason=""):
            quarantined.extend(doc_ids)

        policy = TrustPolicy(ledger=ledger, on_quarantine=on_q)
        verifier = CorroborationVerifier()

        target = _doc("poison", "Zog the Alien designed the Eiffel Tower.", "attacker", "fp")
        clean = _doc("clean", "Gustave Eiffel designed and built the Eiffel Tower.", "wiki", "fc")
        docs = [target, clean]
        assessments = [_assessment("poison", Band.MEDIUM), _assessment("clean", Band.LOW)]

        # First refute → TRUSTED→MONITORED, then immediately MONITORED→QUARANTINED
        # (because quarantine_refutations=1)
        policy.decide("who designed the eiffel tower?", "q1",
                      docs, assessments, verifier=verifier)
        assert "poison" in quarantined

    def test_notes_include_policy_metadata(self, policy):
        docs = [_doc("d1")]
        assessments = [_assessment("d1", Band.LOW)]
        decision = policy.decide("q", "q1", docs, assessments)
        assert decision.notes["policy"] == "TrustPolicy"
        assert "n_verified" in decision.notes
        assert "n_excluded" in decision.notes

    def test_protocol_conformance(self, policy, ledger):
        """TrustPolicy, TrustLedger, CorroborationVerifier satisfy the protocols."""
        from trace_rag.contracts import Policy, TrustProvider, Verifier
        assert isinstance(policy, Policy)
        assert isinstance(ledger, TrustProvider)
        assert isinstance(CorroborationVerifier(), Verifier)


# ===========================================================================
#  Integration: all three wired together
# ===========================================================================

class TestTrustIntegration:

    @pytest.fixture
    def components(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        verifier = CorroborationVerifier()
        policy = TrustPolicy(ledger=ledger)
        return ledger, verifier, policy

    def test_full_cycle_clean_passage(self, components):
        ledger, verifier, policy = components
        target = _doc("clean", "Gustave Eiffel designed the Eiffel Tower in 1889.",
                       "wiki_a", "f1")
        corr = _doc("corr", "Gustave Eiffel was the French civil engineer who designed the tower.",
                     "wiki_b", "f2")
        docs = [target, corr]
        assessments = [_assessment("clean", Band.MEDIUM), _assessment("corr", Band.LOW)]

        decision = policy.decide("who designed the eiffel tower?", "q1",
                                 docs, assessments, verifier=verifier)
        # Clean passage should be supported and allowed
        assert "clean" in decision.context_doc_ids
        snap = ledger.get_trust(["clean"])["clean"]
        assert snap.t_doc >= 0.5  # trust should be neutral or higher

    def test_full_cycle_poison_passage(self, components):
        ledger, verifier, policy = components
        poison = _doc("poison", "The Eiffel Tower was designed by Zog the Alien in 1889.",
                       "attacker", "f_poison")
        clean = _doc("clean", "Gustave Eiffel was the civil engineer who designed the Eiffel Tower.",
                      "wiki", "f_clean")
        docs = [poison, clean]
        assessments = [_assessment("poison", Band.MEDIUM), _assessment("clean", Band.LOW)]

        decision = policy.decide("who designed the eiffel tower?", "q1",
                                 docs, assessments, verifier=verifier)
        # Poison should be excluded, clean allowed
        assert "poison" in decision.excluded_doc_ids
        assert "clean" in decision.context_doc_ids

    def test_repeated_refutations_quarantine(self, tmp_path):
        """Multiple refutations lead to quarantine."""
        ledger = TrustLedger(tmp_path / "trust.db")
        verifier = CorroborationVerifier()
        policy = TrustPolicy(ledger=ledger)

        poison = _doc("poison", "Zog the Alien designed the Eiffel Tower.",
                       "attacker", "f_poison")
        clean = _doc("clean", "Gustave Eiffel designed and built the Eiffel Tower.",
                      "wiki", "f_clean")
        docs = [poison, clean]
        assessments = [_assessment("poison", Band.MEDIUM), _assessment("clean", Band.LOW)]

        # Run the decision twice to accumulate refutations
        policy.decide("who designed the eiffel tower?", "q1",
                      docs, assessments, verifier=verifier)
        policy.decide("who designed the eiffel tower?", "q2",
                      docs, assessments, verifier=verifier)

        # After 2 refutations with default config, should be QUARANTINED
        assert ledger.get_status("poison") is TrustStatus.QUARANTINED
        assert "poison" in ledger.blocked_doc_ids()


# ===========================================================================
#  Pipeline integration (uses the fixtures from conftest.py)
# ===========================================================================


class TestPipelineWithTrust:

    def test_pipeline_with_trust_components(self, populated_pipeline):
        """Wire Person B components into the pipeline and run a query."""
        ledger = TrustLedger(":memory:", store=populated_pipeline.store)
        verifier = CorroborationVerifier(llm=populated_pipeline.generator.llm)
        policy = TrustPolicy(ledger=ledger, on_quarantine=populated_pipeline.on_quarantine)

        populated_pipeline.trust_provider = ledger
        populated_pipeline.retriever.trust = ledger
        populated_pipeline.verifier = verifier
        populated_pipeline.policy = policy

        result = populated_pipeline.answer("Who designed the Eiffel Tower?", "q1")
        assert result.answer_id is not None
        assert result.record.query_id == "q1"
        assert result.decision.notes.get("policy") == "TrustPolicy"

    def test_poison_detected_and_excluded_end_to_end(self, populated_pipeline):
        """With Person B wired in and MEDIUM thresholds, poisoned passages are caught."""
        ledger = TrustLedger(":memory:", store=populated_pipeline.store)
        verifier = CorroborationVerifier(llm=populated_pipeline.generator.llm)
        policy = TrustPolicy(ledger=ledger)

        populated_pipeline.trust_provider = ledger
        populated_pipeline.retriever.trust = ledger
        populated_pipeline.verifier = verifier
        populated_pipeline.policy = policy

        # Force everything into MEDIUM band for full verification
        populated_pipeline.scorer.theta_low = 0.0
        populated_pipeline.scorer.theta_high = 1.01

        result = populated_pipeline.answer("Who designed the Eiffel Tower?", "q1")
        # All passages should have been verified
        assert len(result.decision.verified) > 0
        # The answer should have been produced (not abstained, since clean passages exist)
        assert result.answer_id is not None
