"""Regression tests for the defects found by the code-level audit of real-data-integration."""
from __future__ import annotations

import json
from pathlib import Path

from trace_rag.config import Config
from trace_rag.contracts import Band
from trace_rag.trust.ledger import TrustConfig, TrustLedger
from trace_rag.trust.policy import TrustPolicy

from test_trust import _assessment, _doc


def test_high_band_quarantine_triggers_remediation(tmp_path):
    """A quarantine caused by repeated HIGH-band penalties must flag earlier answers."""
    cfg = TrustConfig(quarantine_refutations=99, monitored_t_eff=0.95,
                      quarantine_t_eff=0.90, high_band_beta_penalty=3.0)
    ledger = TrustLedger(tmp_path / "t.db", config=cfg)
    flagged = []
    policy = TrustPolicy(ledger=ledger, on_quarantine=lambda ids, reason="": flagged.extend(ids))
    doc = _doc("poison", "Zog the Alien designed the Eiffel Tower.", "attacker", "fp")
    for i in range(6):
        decision = policy.decide("who designed it?", f"q{i}", [doc], [_assessment("poison", Band.HIGH)])
        if "poison" in ledger.blocked_doc_ids():
            break
    assert "poison" in ledger.blocked_doc_ids(), "setup: penalties should have quarantined it"
    assert flagged == ["poison"], "HIGH-band quarantine must call the remediation callback once"
    assert "poison" in decision.notes["newly_quarantined"]


def test_record_high_band_reports_the_transition_only_once(tmp_path):
    cfg = TrustConfig(quarantine_refutations=99, monitored_t_eff=0.95,
                      quarantine_t_eff=0.90, high_band_beta_penalty=3.0)
    ledger = TrustLedger(tmp_path / "t.db", config=cfg)
    results = [ledger.record_high_band("d", "s", "f") for _ in range(6)]
    assert results.count(True) == 1
    assert ledger.record_high_band("d", "s", "f") is False   # already blocked: not "new"


def test_baseline_verifier_matches_the_ragledger_verifier():
    """Same security settings -> same verifier mode and thresholds in both arms."""
    from trace_rag.evaluation.experiment import build_baseline_verifier
    from trace_rag.pipeline import PersonAPipeline

    config = Config.load(None, security={"verifier": "lexical", "support_threshold": 0.77,
                                         "refute_threshold": 0.66, "max_source_mass": 0.4})
    secure = PersonAPipeline.from_config(config, load_existing_index=False, secure=True)
    plain = PersonAPipeline.from_config(config, load_existing_index=False, secure=False)
    ours, theirs = secure.verifier, build_baseline_verifier(plain, config)
    assert ours.mode == theirs.mode == "lexical"
    for attr in ("support_threshold", "refute_threshold", "max_source_mass",
                 "counterfactual_influence"):
        assert getattr(ours, attr) == getattr(theirs, attr), attr
    assert theirs.support_threshold == 0.77


def _tiny_fixture():
    from trace_rag.evaluation.experiment import load_fixture
    return load_fixture(Path(__file__).resolve().parent.parent / "examples")


def test_signature_changes_with_config_overrides_and_extras():
    from trace_rag.evaluation.signature import run_signature
    fx = _tiny_fixture()
    base = run_signature(fx, config_path=None, overrides=[], target_limit=2, repeat_targets=2)
    assert base["digest"] == run_signature(fx, config_path=None, overrides=[], target_limit=2,
                                           repeat_targets=2)["digest"]
    for kwargs in ({"overrides": ["security.verifier=nli"]}, {"target_limit": 3},
                   {"extra": {"generator_digest": "sha256:abc"}}):
        args = dict(config_path=None, overrides=[], target_limit=2, repeat_targets=2)
        args.update(kwargs)
        assert run_signature(fx, **args)["digest"] != base["digest"], kwargs


def test_resume_refuses_cases_from_a_different_signature(tmp_path):
    from trace_rag.evaluation import experiment as ex
    fx = _tiny_fixture()
    out = tmp_path / "out"
    kw = dict(target_limit=2, scope_label="plumbing", resume=True)
    ex.run_matrix(["hit_and_run"], ["no_defence"], [1], fx, out, signature_extra={"gen": "A"}, **kw)
    result_file = out / "cases" / "hit_and_run__no_defence__seed1" / "result.json"
    stamp = result_file.stat().st_mtime_ns
    ex.run_matrix(["hit_and_run"], ["no_defence"], [1], fx, out, signature_extra={"gen": "A"}, **kw)
    assert result_file.stat().st_mtime_ns == stamp, "same signature: reuse"
    ex.run_matrix(["hit_and_run"], ["no_defence"], [1], fx, out, signature_extra={"gen": "B"}, **kw)
    assert result_file.stat().st_mtime_ns != stamp, "different model digest: must recompute"
    data = json.loads(result_file.read_text())
    assert data["_signature"] and data["_complete"]


def test_attacker_string_is_recorded_and_audit_sample_is_stratified(tmp_path):
    """The audit sampler must separate counted ASR successes from quoted/uncounted attacker strings."""
    from trace_rag.evaluation import experiment as ex
    from trace_rag.evaluation.analysis import audit_sample

    fx = _tiny_fixture()
    out = tmp_path / "out"
    ex.run_matrix(["hit_and_run"], ["no_defence"], [1], fx, out, target_limit=2,
                  scope_label="plumbing")
    stream = next((out / "cases").glob("*/stream.jsonl"))
    rows = [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
    targeted = [r for r in rows if r.get("poison_targeted")]
    assert targeted and all("attacker_string_present" in r and "malicious_answer" in r for r in targeted)
    sample = audit_sample(out, per_stratum=3)
    assert sample and all(r["human_label"] == "" for r in sample)
    assert {r["stratum"] for r in sample} <= {"asr_success", "attacker_string_not_counted",
                                                "gold_correct", "neither", "abstained"}
