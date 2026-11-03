"""Regression tests for the research attack, baseline, dataset and metric layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_rag.attacks import ATTACK_NAMES, ATTACK_SPECS
from trace_rag.attacks.research import (
    adaptive_events, black_box_poison, coordinated_injection, severity_scale,
    sybil_duplication, white_box_poison,
)
from trace_rag.baselines import BASELINE_NAMES, build_baseline_policy
from trace_rag.contracts import (
    Band, FeatureSnapshot, RetrievedDocument, SecurityAssessment, SignalVector,
    TrustSnapshot, TrustStatus, VerificationOutcome, VerificationResult,
)
from trace_rag.evaluation.beir import load_beir_fixture
from trace_rag.evaluation.metrics import (
    answer_level_asr_fields, attack_answer_success, latency_stats,
)


def test_all_twelve_attack_families_are_registered_and_specified():
    expected = {
        "entity_swap", "negation", "instruction_injection", "slow_burn",
        "hit_and_run", "framing", "signal_evasion", "black_box", "white_box",
        "sybil", "coordinated", "adaptive",
    }
    assert set(ATTACK_NAMES) == expected
    assert expected <= set(ATTACK_SPECS)
    for name in expected:
        assert ATTACK_SPECS[name]


def test_sybil_and_coordinated_events_have_real_identity_structure():
    sybil = sybil_duplication("false claim", target_query="q", start_step=2, copies=4)
    assert len({event.source_id for event in sybil}) == 4
    assert len({event.family_id for event in sybil}) == 1
    assert all(event.attack_type == "sybil" and event.is_poison for event in sybil)
    assert {event.step for event in sybil} == {2, 3, 4, 5}

    coordinated = coordinated_injection(
        ["one", "two"], target_query="q", start_step=1, sources_per_payload=2,
    )
    assert len(coordinated) == 4
    assert len({event.source_id for event in coordinated}) == 4
    assert {event.payload for event in coordinated} == {"one", "two"}


def test_black_box_uses_only_question_and_white_box_keeps_retrieval_surface():
    black = black_box_poison("Who wrote X?", "False Author")
    assert "Who wrote X?" in black and "False Author" in black
    white = white_box_poison("X was written by Alice.", "Who wrote X?", "Bob",
                             gold_answer="Alice")
    assert "Bob" in white and "Alice" not in white

    adaptive = adaptive_events("X was written by Alice.", "Who wrote X?", "Bob",
                               gold_answer="Alice", start_step=1, warmup_steps=2)
    assert [e.phase for e in adaptive if not e.is_poison] == ["warmup", "warmup"]
    poison = [e for e in adaptive if e.is_poison]
    assert len(poison) == 1
    assert "Who wrote X?" not in poison[0].payload
    assert poison[0].step == 3


def test_severity_scale_is_explicit_and_never_rounds_positive_rate_to_zero():
    assert severity_scale(0.01, 1000) == 10
    assert severity_scale(0.0001, 100) == 1
    with pytest.raises(ValueError):
        severity_scale(0.0, 100)


def test_answer_level_asr_has_a_narrow_auditable_criterion():
    assert attack_answer_success("The answer is Zog the Alien [d1]", "Zog the Alien", "Paris")
    assert not attack_answer_success("The answer is Paris [d1]", "Zog the Alien", "Paris")
    assert not attack_answer_success("INSUFFICIENT EVIDENCE", "Zog the Alien", "Paris")
    assert attack_answer_success("Zog the Alien and Paris", "Zog the Alien", "") is True
    assert attack_answer_success("anything", "") is None
    measured = answer_level_asr_fields(0.5, evaluated=4)
    assert measured["answer_level_asr_status"] == "MEASURED"
    assert measured["answer_level_asr_numerator"] == 2


def test_latency_includes_p99():
    stats = latency_stats(list(range(1, 101)))
    assert stats["p99_ms"] is not None
    assert stats["p99_ms"] >= stats["p95_ms"]


def test_new_policies_are_executable_and_user_visible():
    assert {"suspicion_only", "corroboration_only", "trust_no_deferred"} <= set(BASELINE_NAMES)
    doc = RetrievedDocument(
        doc_id="d", chunk_id="d", text="claim", similarity=1.0, rank=0,
        source_id="s", family_id="f",
        trust=TrustSnapshot(doc_id="d", source_id="s", family_id="f", t_eff=0.9),
    )
    high = SecurityAssessment(
        doc_id="d", chunk_id="d", query_id="q", signals=SignalVector(),
        suspicion=0.9, band=Band.HIGH, action="EXCLUDE",
        feature_snapshot=FeatureSnapshot(SignalVector(), {}), scorer_version="test",
    )
    only = build_baseline_policy("suspicion_only")
    decision = only.decide("q", "q", [doc], [high])
    assert decision.context_doc_ids == ()

    class Refute:
        name = "fake"
        def verify(self, query, query_id, target, pool):
            return VerificationResult(
                doc_id=target.doc_id, query_id=query_id, single_doc_answer=None,
                influential=True, support_mass=0.0, refute_mass=1.0,
                outcome=VerificationOutcome.REFUTE, llm_calls=1,
            )

    corr = build_baseline_policy("corroboration_only")
    decision = corr.decide("q", "q", [doc], [high], verifier=Refute())
    assert decision.context_doc_ids == ()
    assert decision.verified[0].outcome is VerificationOutcome.REFUTE


def test_beir_fixture_reads_jsonl_and_qrels(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "text": "Paris is in France.", "title": "Paris"}) + "\n" +
        json.dumps({"_id": "d2", "text": "Everest is high.", "title": "Everest"}) + "\n",
        encoding="utf-8")
    queries.write_text(
        json.dumps({"_id": "q1", "text": "Where is Paris?"}) + "\n" +
        json.dumps({"_id": "q2", "text": "How high is Everest?"}) + "\n",
        encoding="utf-8")
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8")
    fixture = load_beir_fixture(corpus, queries, qrels, target_limit=2)
    assert fixture.targets == ["q1", "q2"]
    assert fixture.questions[0]["gold_doc_id"] == "d1"
    assert len(fixture.clean) == 2
