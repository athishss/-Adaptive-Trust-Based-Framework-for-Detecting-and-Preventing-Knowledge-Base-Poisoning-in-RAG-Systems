"""Tests for the Person B plan gaps: B1 corrections, B2 Sybil defence, B3
verification budget and influence, B4 queue and administrator review, B6
ablation switch and trust-dynamics plots.

Every test here corresponds to a clause of the revised plan the previous
implementation did not honour; the docstrings name the clause.
"""

from __future__ import annotations

import time

import pytest

from trace_rag.contracts import (
    Band,
    FeatureSnapshot,
    RetrievedDocument,
    SecurityAssessment,
    SignalVector,
    TrustSnapshot,
    TrustStatus,
    VerificationOutcome,
    VerificationResult,
)
from trace_rag.generation.llm import LLMResponse, StubLLM
from trace_rag.trust.ledger import TrustConfig, TrustLedger
from trace_rag.trust.plots import (
    plot_quarantine_timeline,
    plot_trust_dynamics,
    write_history_csv,
)
from trace_rag.trust.policy import TrustPolicy
from trace_rag.trust.queue import VerificationQueue
from trace_rag.trust.verifier import CorroborationVerifier

CLAIM = "The Zog artefact was discovered by Maria Chen in 1998."
QUERY = "who discovered the Zog artefact?"
REPEAT = CLAIM
RIVAL = "The Zog artefact was discovered by Alan Wu in 1999."


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _doc(doc_id: str, text: str = "some text", source_id: str = "s",
         family_id: str = "f", t_source: float = 0.5,
         t_eff: float = 0.5) -> RetrievedDocument:
    snap = TrustSnapshot(doc_id=doc_id, source_id=source_id, family_id=family_id,
                         t_doc=t_eff, t_family=t_eff, t_source=t_source,
                         t_eff=t_eff, status=TrustStatus.TRUSTED)
    return RetrievedDocument(doc_id=doc_id, chunk_id=doc_id, text=text,
                             similarity=0.8, rank=0, source_id=source_id,
                             family_id=family_id, trust=snap,
                             metadata={"ingested_at": 1.0})


def _assessment(doc_id: str, band: Band = Band.LOW) -> SecurityAssessment:
    return SecurityAssessment(
        doc_id=doc_id, chunk_id=doc_id, query_id="q", signals=SignalVector(),
        suspicion=0.1, band=band, action=band.value,
        feature_snapshot=FeatureSnapshot(SignalVector(), {}),
        scorer_version="test",
    )


class EchoLLM:
    """Returns a fixed text for every prompt (deterministic counterfactuals)."""

    name = "echo"

    def __init__(self, text: str = CLAIM) -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompt, max_tokens=256, stop=None):  # noqa: ARG002
        self.calls += 1
        return LLMResponse(text=self.text, llm_calls=1)


class FixedVerifier:
    """Minimal Verifier protocol implementation: one fixed outcome."""

    def __init__(self, outcome: VerificationOutcome = VerificationOutcome.REFUTE):
        self.outcome = outcome
        self.calls = 0

    def verify(self, query, query_id, target, pool):  # noqa: ARG002
        self.calls += 1
        return VerificationResult(
            doc_id=target.doc_id, query_id=query_id, single_doc_answer="x",
            influential=True, support_mass=0.0, refute_mass=0.9,
            outcome=self.outcome, llm_calls=0, latency_ms=1.0,
        )


# ===========================================================================
#  B1 - ledger corrections
# ===========================================================================

class TestB1LedgerCorrections:

    def test_parent_trust_uses_worst_of_source_and_family(self):
        """Plan Section 4.6: the parent term is min(T_source, T_family).

        Averaging the two let a damaged family hide behind a clean source.
        """
        ledger = TrustLedger(":memory:")
        # Damage one family badly while the source stays clean ...
        for _ in range(3):
            ledger.record_observation("bad", "clean_src", "bad_fam", "REFUTE",
                                      timestamp=1.0)
        for _ in range(6):
            ledger.record_observation("good", "clean_src", "good_fam", "SUPPORT",
                                      timestamp=1.0)
        ledger.record_observation("sib", "clean_src", "bad_fam", "NEUTRAL",
                                  timestamp=1.0)
        snap = ledger.get_trust(["sib"])["sib"]

        assert snap.t_family < snap.t_source, "fixture should split the two levels"
        w_doc = 1.0 / 5.0                      # one NEUTRAL observation, m = 5
        averaged_parents = 0.5 * (snap.t_family + snap.t_source)
        old_blend = w_doc * snap.t_doc + (1.0 - w_doc) * averaged_parents
        assert snap.t_eff < old_blend - 0.05, (
            "t_eff must use min(T_source, T_family), not their average"
        )

    def test_document_only_ablation_has_no_hierarchy(self):
        """B6/V2: with hierarchical=False nothing is written to family/source."""
        ledger = TrustLedger(":memory:", config=TrustConfig(hierarchical=False))
        for _ in range(3):
            ledger.record_observation("bad", "src", "fam", "REFUTE", timestamp=1.0)
        ledger.record_observation("sib", "src", "fam", "NEUTRAL", timestamp=1.0)
        rows = ledger._conn.execute(
            "SELECT COUNT(*) AS n FROM trust_entities WHERE entity_type IN ('source', 'family')"
        ).fetchone()["n"]
        assert rows == 0
        snap = ledger.get_trust(["sib"])["sib"]
        assert snap.t_eff == pytest.approx(0.5), \
            "document-only mode must not inherit family/source damage"

    def test_decay_is_scheduled_per_query_not_per_observation(self):
        """Plan Section 4.6: forgetting happens every N queries."""
        ledger = TrustLedger(":memory:",
                             config=TrustConfig(decay_gamma=0.5, decay_interval=2))
        ledger.record_observation("d1", "s1", "f1", "SUPPORT", timestamp=1.0)
        before = ledger.get_trust(["d1"])["d1"].t_doc

        # Ten observations from another document: no decay, those are not queries.
        for i in range(10):
            ledger.record_observation(f"other{i}", f"src{i}", f"fam{i}", "SUPPORT",
                                      timestamp=2.0)
        assert ledger.get_trust(["d1"])["d1"].t_doc == pytest.approx(before)

        ledger.begin_query(timestamp=3.0)          # query 1 - no cycle yet
        assert ledger.get_trust(["d1"])["d1"].t_doc == pytest.approx(before)
        ledger.begin_query(timestamp=4.0)          # query 2 - decay fires
        assert ledger.get_trust(["d1"])["d1"].t_doc < before


# ===========================================================================
#  B2 - cold start and Sybil defence
# ===========================================================================

class TestB2SybilDefence:

    def test_burst_detected_from_registration_history(self):
        """Plan signal S4: several documents from one source in a short window."""
        ledger = TrustLedger(":memory:")
        for i in range(3):
            ledger.record_observation(f"d{i}", "attacker", "f1", "NEUTRAL",
                                      timestamp=1000.0)
        assert not ledger._is_burst_source("attacker"), "3 docs is not a burst"
        ledger.record_observation("d3", "attacker", "f1", "NEUTRAL", timestamp=1000.0)
        assert ledger._is_burst_source("attacker"), "4 docs in the window is a burst"

    def test_burst_discount_applies_to_the_earlier_documents_too(self):
        """A burst is only visible once it has happened; the prior still drops."""
        ledger = TrustLedger(":memory:")
        for i in range(4):
            ledger.record_observation(f"b{i}", "attacker", "f1", "NEUTRAL",
                                      timestamp=1000.0)
        ledger.record_observation("clean", "normal", "f2", "NEUTRAL",
                                  timestamp=1000.0)
        burst = ledger.get_trust(["b0"])["b0"]
        normal = ledger.get_trust(["clean"])["clean"]
        assert burst.t_doc < normal.t_doc, \
            "first documents of a burst must carry the discounted prior"

    def test_same_burst_needs_a_shared_group(self):
        ledger = TrustLedger(":memory:")
        for i in range(4):
            ledger.record_observation(f"a{i}", "coordinated_1", "f1", "NEUTRAL",
                                      timestamp=1000.0)
        for i in range(4):
            ledger.record_observation(f"b{i}", "coordinated_2", "f2", "NEUTRAL",
                                      timestamp=1005.0)
        ledger.record_observation("c0", "quiet", "f3", "NEUTRAL", timestamp=1010.0)
        assert ledger.same_burst("coordinated_1", "coordinated_2")
        assert not ledger.same_burst("coordinated_1", "quiet")
        assert not ledger.same_burst("quiet", "coordinated_2")

    def test_new_source_influence_ramps_with_age_or_history(self):
        """Plan Section 4.6: new sources start neutral, their influence is capped."""
        cfg = TrustConfig(source_age_ramp_hours=24.0, cold_start_influence=0.25,
                          cold_start_min_observations=3)
        ledger = TrustLedger(":memory:", config=cfg)
        assert ledger.influence_factor("never_seen", now=1000.0) == pytest.approx(0.25)

        ledger.record_observation("d1", "young", "f1", "NEUTRAL", timestamp=1000.0)
        assert ledger.influence_factor("young", now=1000.0) == pytest.approx(0.25)
        # Halfway up the age ramp ...
        assert ledger.influence_factor("young", now=1000.0 + 12 * 3600) == pytest.approx(
            0.625)
        # ... or with enough verified observations behind it.
        for _ in range(3):
            ledger.record_observation("d1", "young", "f1", "SUPPORT", timestamp=1000.0)
        assert ledger.influence_factor("young", now=1000.0) == pytest.approx(1.0)

    def test_full_age_ramp_reaches_full_influence(self):
        ledger = TrustLedger(":memory:")
        ledger.record_observation("d1", "src", "f1", "NEUTRAL", timestamp=1000.0)
        assert ledger.influence_factor("src", now=1000.0 + 24 * 3600) == pytest.approx(1.0)

    def test_document_only_mode_has_no_influence_cap(self):
        ledger = TrustLedger(":memory:", config=TrustConfig(hierarchical=False))
        assert ledger.influence_factor("unknown_src", now=1000.0) == 1.0


# ===========================================================================
#  B3 - verifier: influence, budget, source weighting
# ===========================================================================

class TestB3Verifier:

    def _verifier(self, **kwargs) -> CorroborationVerifier:
        kwargs.setdefault("use_nli", False)
        kwargs.setdefault("llm", StubLLM())
        return CorroborationVerifier(**kwargs)

    def test_independent_filter_drops_the_same_burst(self):
        """Plan Section 4.5: corroborators must not come from the same burst."""
        verifier = self._verifier()
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target, _doc("t2", REPEAT, "attacker_sibling", "f2")]
        assert verifier._find_independent(target, pool) != []
        assert verifier._find_independent(
            target, pool, lambda a, b: True) == []

    def test_same_burst_context_is_used_by_verify(self):
        verifier = self._verifier()
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target, _doc("t2", RIVAL, "attacker_sibling", "f2", t_source=0.9)]
        without = verifier.verify(QUERY, "q1", target, pool)
        with_burst = verifier.verify(QUERY, "q1", target, pool,
                                     same_burst=lambda a, b: True)
        assert with_burst.outcome is VerificationOutcome.NEUTRAL
        assert without.outcome is not VerificationOutcome.NEUTRAL

    def test_mass_uses_source_trust_and_is_capped(self):
        """Plan Section 4.5: mass = sum T_source(d'); no source may dominate."""
        target = _doc("t1", CLAIM, "attacker", "f1")
        corroborator = _doc("t2", REPEAT, "clean", "f2", t_source=0.95)
        pool = [target, corroborator]

        capped = self._verifier(max_source_mass=0.6).verify(QUERY, "q1", target, pool)
        uncapped = self._verifier(max_source_mass=1.0).verify(QUERY, "q1", target, pool)
        assert 0.0 < capped.support_mass < uncapped.support_mass
        assert capped.support_mass <= 0.6 + 1e-9

    def test_cold_start_influence_scales_the_mass(self):
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target, _doc("t2", REPEAT, "clean", "f2", t_source=0.5)]
        verifier = self._verifier()
        full = verifier.verify(QUERY, "q1", target, pool)
        quartered = verifier.verify(QUERY, "q1", target, pool,
                                    source_influence=lambda src: 0.25)
        assert quartered.support_mass < full.support_mass
        assert quartered.support_mass == pytest.approx(full.support_mass / 4.0, rel=0.05)

    def test_lexical_path_needs_no_extra_llm_calls(self):
        """Plan Section 4.5 budget: claim extraction + influence, nothing per passage."""
        llm = EchoLLM(CLAIM)
        verifier = CorroborationVerifier(llm=llm, use_nli=False,
                                        counterfactual_influence=False)
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target] + [_doc(f"p{i}", RIVAL, f"s{i}", f"f{i}", t_source=0.9)
                           for i in range(4)]
        result = verifier.verify(QUERY, "q1", target, pool)
        assert result.llm_calls == 1, \
            "the lexical path must not extract a claim per independent passage"
        assert llm.calls == 1

    def test_total_budget_is_at_most_two_calls(self):
        """Claim extraction plus one influence call, whatever the pool size."""
        verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False,
                                        counterfactual_influence=True)
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target] + [_doc(f"p{i}", RIVAL, f"s{i}", f"f{i}", t_source=0.9)
                           for i in range(5)]
        result = verifier.verify(QUERY, "q1", target, pool)
        assert result.llm_calls == 2

    def test_counterfactual_influence_gates_refutation(self):
        """Plan Section 4.5: REFUTE needs an influential passage.

        If removing the passage leaves the answer unchanged, it cannot be the
        thing poisoning the answer - even when independent sources contradict it.
        """
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target, _doc("t2", RIVAL, "clean", "f2", t_source=0.9)]

        gated = CorroborationVerifier(llm=EchoLLM(CLAIM), use_nli=False,
                                      counterfactual_influence=True)
        result = gated.verify(QUERY, "q1", target, pool)
        assert result.refute_mass > 0.0, "the contradiction must still be measured"
        assert result.influential is False
        assert result.outcome is VerificationOutcome.NEUTRAL

        ungated = CorroborationVerifier(llm=EchoLLM(CLAIM), use_nli=False,
                                        counterfactual_influence=False)
        assert ungated.verify(QUERY, "q1", target, pool).outcome is \
            VerificationOutcome.REFUTE

    def test_non_influential_passage_can_still_be_supported(self):
        """The influence gate must not block SUPPORT (plan Section 4.5)."""
        verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False,
                                        counterfactual_influence=True)
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target, _doc("t2", REPEAT, "clean", "f2", t_source=0.9)]
        result = verifier.verify(QUERY, "q1", target, pool)
        assert result.influential is False       # repeated verbatim elsewhere
        assert result.outcome is VerificationOutcome.SUPPORT

    def test_counterfactual_detects_a_changed_answer(self):
        """A passage whose removal changes the answer stays influential."""
        verifier = CorroborationVerifier(llm=StubLLM(), use_nli=False,
                                        counterfactual_influence=True)
        target = _doc("t1", CLAIM, "attacker", "f1")
        pool = [target, _doc("t2", RIVAL, "clean", "f2", t_source=0.9)]
        result = verifier.verify(QUERY, "q1", target, pool)
        assert result.influential is True
        assert result.outcome is VerificationOutcome.REFUTE


# ===========================================================================
#  B4 - async queue and administrator review
# ===========================================================================

class TestB4Queue:

    def test_high_band_is_queued_and_drained_off_path(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        queue = VerificationQueue()
        policy = TrustPolicy(ledger=ledger, queue=queue)
        verifier = FixedVerifier(VerificationOutcome.REFUTE)
        docs = [_doc("d1", CLAIM, "attacker", "f1")]
        assessments = [_assessment("d1", Band.HIGH)]

        decision = policy.decide(QUERY, "q1", docs, assessments, verifier=verifier)
        assert "d1" in decision.excluded_doc_ids      # excluded now ...
        assert len(queue) == 1                        # ... verified later
        assert verifier.calls == 0                    # nothing on the latency path

        results = policy.drain_verification_queue(verifier)
        assert verifier.calls == 1
        assert [r.doc_id for r in results] == ["d1"]
        # One observation from the HIGH-band penalty, one from the drained
        # verification: the queued outcome is recorded exactly like an inline one.
        assert ledger.get_trust(["d1"])["d1"].n_doc_observations == 2
        assert len(queue) == 0

    def test_queue_dedupes_by_doc_id(self):
        queue = VerificationQueue()
        doc = _doc("d1")
        first = queue.enqueue(QUERY, "q1", doc, [doc])
        second = queue.enqueue(QUERY, "q2", doc, [doc])
        assert first is True and second is False
        assert len(queue) == 1

    def test_queue_is_bounded(self):
        queue = VerificationQueue(max_size=2)
        for i in range(3):
            queue.enqueue(QUERY, f"q{i}", _doc(f"d{i}"), [])
        assert len(queue) == 2
        assert queue.dropped == 1

    def test_background_worker_drains(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        queue = VerificationQueue()
        verifier = FixedVerifier(VerificationOutcome.SUPPORT)
        queue.enqueue(QUERY, "q1", _doc("d1"), [])
        queue.start_background(verifier, ledger, interval=0.05)
        deadline = time.monotonic() + 5.0
        try:
            while len(queue) and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            queue.stop_background()
        assert verifier.calls == 1
        assert ledger.get_trust(["d1"])["d1"].n_doc_observations == 1


class TestB4AdminReview:

    def test_review_is_raised_instead_of_auto_rejection(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        for _ in range(3):
            ledger.record_observation("d1", "src", "fam", "REFUTE", timestamp=1.0)
        assert ledger.get_status("d1") is TrustStatus.QUARANTINED
        pending = [r["entity_id"] for r in ledger.pending_reviews()]
        assert pending == ["d1"]

    def test_decline_keeps_the_document_quarantined(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        for _ in range(3):
            ledger.record_observation("d1", "src", "fam", "REFUTE", timestamp=1.0)
        assert ledger.admin_decline_rejection("d1", reviewed_by="admin") is True
        assert ledger.pending_reviews() == []
        assert ledger.get_status("d1") is TrustStatus.QUARANTINED

    def test_administrator_recovery_is_the_only_way_out(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        ledger.record_observation("d1", "src", "fam", "REFUTE", timestamp=1.0)
        ledger.record_observation("d1", "src", "fam", "REFUTE", timestamp=1.0)
        assert ledger.get_status("d1") is TrustStatus.QUARANTINED
        for _ in range(10):
            ledger.record_observation("d1", "src", "fam", "SUPPORT", timestamp=1.0)
        assert ledger.get_status("d1") is TrustStatus.QUARANTINED, \
            "support alone must not talk a document back into use"
        assert ledger.admin_approve_recovery("d1", approved_by="admin") is True
        assert ledger.get_status("d1") is TrustStatus.MONITORED
        assert "d1" not in ledger.blocked_doc_ids()

    def test_rejection_flags_the_source(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        for _ in range(3):
            ledger.record_observation("d1", "attacker", "fam", "REFUTE", timestamp=1.0)
        before = ledger.get_trust(["d1"])["d1"].t_source
        assert ledger.admin_approve_rejection("d1", approved_by="admin",
                                              note="confirmed") is True
        after = ledger.get_trust(["d1"])["d1"].t_source
        assert after < before
        log = ledger.get_audit_log("d1")
        assert any(e["trigger"] == "ADMIN_APPROVAL" and e["new_status"] == "REJECTED"
                   for e in log)

    def test_cannot_reject_a_document_that_is_not_quarantined(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db")
        ledger.record_observation("d1", "src", "fam", "SUPPORT", timestamp=1.0)
        assert ledger.admin_approve_rejection("d1", approved_by="admin") is False


# ===========================================================================
#  B6 - trust history and plots
# ===========================================================================

class TestB6TrustDynamics:

    def _populated(self, tmp_path) -> TrustLedger:
        ledger = TrustLedger(tmp_path / "trust.db")
        ledger.record_observation("d1", "src", "fam", "SUPPORT", timestamp=1.0)
        ledger.record_observation("d2", "src", "fam", "REFUTE", timestamp=2.0)
        ledger.record_observation("d2", "src", "fam", "REFUTE", timestamp=3.0)
        return ledger

    def test_history_records_every_update(self, tmp_path):
        history = self._populated(tmp_path).export_history()
        assert history, "trust history must be written by default"
        assert {"doc", "source"} <= {row["entity_type"] for row in history}
        assert all(0.0 <= row["trust"] <= 1.0 for row in history)

    def test_history_can_be_switched_off(self, tmp_path):
        ledger = TrustLedger(tmp_path / "trust.db",
                             config=TrustConfig(record_history=False))
        ledger.record_observation("d1", "src", "fam", "SUPPORT", timestamp=1.0)
        assert ledger.export_history() == []

    def test_export_history_filters(self, tmp_path):
        ledger = self._populated(tmp_path)
        only_d1 = ledger.export_history(entity_id="d1")
        assert only_d1 and all(row["entity_id"] == "d1" for row in only_d1)

    def test_history_csv_is_plot_ready(self, tmp_path):
        ledger = self._populated(tmp_path)
        path = write_history_csv(ledger.export_history(), tmp_path / "history.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("timestamp,entity_id,entity_type,trust")
        assert len(lines) > 1

    def test_trust_dynamics_plot_is_written(self, tmp_path):
        pytest.importorskip("matplotlib")
        ledger = self._populated(tmp_path)
        files = plot_trust_dynamics(ledger.export_history(), tmp_path / "plots")
        assert files, "one figure per entity is expected"
        assert all(path.exists() and path.stat().st_size > 0 for path in files)

    def test_quarantine_timeline_plot(self, tmp_path):
        pytest.importorskip("matplotlib")
        ledger = self._populated(tmp_path)
        path = plot_quarantine_timeline(ledger.get_audit_log(), tmp_path / "blocked.png")
        assert path is not None and path.exists()

    def test_empty_history_plots_nothing(self, tmp_path):
        pytest.importorskip("matplotlib")
        assert plot_trust_dynamics([], tmp_path / "plots") == []
        assert plot_quarantine_timeline([], tmp_path / "blocked.png") is None
