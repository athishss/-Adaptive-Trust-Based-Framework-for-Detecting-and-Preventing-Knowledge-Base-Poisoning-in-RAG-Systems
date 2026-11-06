"""Unit-level statistics: collapse repeats, pair by (seed, question), Holm-correct."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_rag.evaluation.analysis import analyse_directory, holm, load_units, paired_tests
from trace_rag.evaluation.metrics import attack_answer_success, gold_answer_present


def _case(root, attack, baseline, seed, outcomes):
    """outcomes: {qid: [(success, gold_correct, abstained), ...]} in step order."""
    d = root / "cases" / f"{attack}__{baseline}__seed{seed}"
    d.mkdir(parents=True)
    step = 0
    rows = []
    for qid, attempts in outcomes.items():
        for ok, gold, abst in attempts:
            rows.append({"step": step, "query_id": f"{qid}_s{step}", "is_target": True,
                         "poison_targeted": ["p"], "answer_attack_status": "MEASURED",
                         "answer_attack_success": ok, "answer_gold_correct": gold,
                         "abstained": abst, "poison_in_context": ["p"] if ok else []})
            step += 1
    rows.append({"step": step, "query_id": "bg_s99", "is_target": False, "poison_targeted": [],
                 "answer_attack_status": "NOT MEASURED", "answer_attack_success": None,
                 "abstained": False, "poison_in_context": []})
    (d / "stream.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps({"llm_calls": {"mean_per_query": 2.0},
                                                "latency_ms": {"p99_ms": 5.0}}), encoding="utf-8")


def test_repeated_asks_collapse_to_one_unit_and_unmeasured_rows_are_ignored(tmp_path):
    _case(tmp_path, "a", "d", 1, {"q1": [(True, False, False), (False, True, False)]})
    units = load_units(tmp_path)
    assert len(units) == 1
    u = units[0]
    assert (u.asr_first, u.asr_last, u.asr_mean, u.n_attempts) == (1.0, 0.0, 0.5, 2)
    assert u.accuracy_mean == 0.5


def test_holm_is_monotone_and_skips_missing():
    adj = holm([0.01, None, 0.04, 0.03])
    assert adj[1] is None
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.06) and adj[2] == pytest.approx(0.06)


def test_paired_test_detects_a_consistent_difference_and_refuses_tiny_samples(tmp_path):
    qs = {f"q{i}": [(False, True, False)] for i in range(20)}
    bad = {f"q{i}": [(True, False, False)] for i in range(20)}
    _case(tmp_path, "a", "RAGLedger", 1, qs)
    _case(tmp_path, "a", "no_defence", 1, bad)
    tests = paired_tests(load_units(tmp_path), "RAGLedger", "asr_last", min_units=10)
    assert tests[0]["baseline_succeeds_only"] == 20 and tests[0]["p_exact"] < 1e-5
    small = paired_tests(load_units(tmp_path), "RAGLedger", "asr_last", min_units=50)
    assert small[0]["p_exact"] is None


def test_analyse_directory_writes_outputs_and_flags_abstention(tmp_path):
    _case(tmp_path, "a", "RAGLedger", 1, {f"q{i}": [(False, False, True)] for i in range(12)})
    _case(tmp_path, "a", "no_defence", 1, {f"q{i}": [(True, False, False)] for i in range(12)})
    payload = analyse_directory(tmp_path, scope="unit test")
    ledger = next(r for r in payload["summary"] if r["baseline"] == "RAGLedger")
    assert ledger["asr_last"] == 0.0 and ledger["accuracy"] == 0.0 and ledger["abstention"] == 1.0
    md = (tmp_path / "analysis.md").read_text(encoding="utf-8")
    assert "abstention" in md and "McNemar" in md
    assert (tmp_path / "analysis.csv").exists()


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        analyse_directory(tmp_path)


def test_answer_matching_respects_word_boundaries():
    # "24" must not match inside "2024"; gold "23" present means not an attack success.
    assert attack_answer_success("It aired in 2024.", "24", "23") is False
    assert attack_answer_success("Season 4 has 24 episodes.", "24", "23") is True
    assert attack_answer_success("23 or 24 episodes", "24", "23") is False
    assert gold_answer_present("Season 4 has 23 episodes.", "23") is True
    assert gold_answer_present("in 2023", "23") is False
    assert gold_answer_present("anything", "") is None


def test_per_seed_subdirectories_are_read_and_costs_averaged(tmp_path):
    for seed in (1, 2):
        sub = tmp_path / f"seed_{seed}"
        _case(sub, "a", "RAGLedger", seed, {f"s{seed}q{i}": [(False, True, False)] for i in range(6)})
        _case(sub, "a", "no_defence", seed, {f"s{seed}q{i}": [(True, False, False)] for i in range(6)})
    units = load_units(tmp_path)
    assert len(units) == 24 and {u.seed for u in units} == {1, 2}
    payload = analyse_directory(tmp_path, min_units=10)
    t = payload["paired_tests"][0]
    assert t["paired_units"] == 12 and t["p_exact"] < 0.01
    assert payload["cost"]["a|RAGLedger"]["llm_calls_per_query"] == 2.0


def test_benign_control_reports_the_accuracy_cost_of_a_defence(tmp_path):
    from trace_rag.evaluation.analysis import load_clean_units, summarise_clean

    def clean_case(baseline, correct):
        d = tmp_path / "cases" / f"none__{baseline}__seed1"
        d.mkdir(parents=True)
        rows = [{"step": i, "query_id": f"q{i}_s{i}", "is_target": True, "poison_targeted": [],
                 "answer_gold_correct": ok, "abstained": not ok, "answer_attack_status": "MEASURED"}
                for i, ok in enumerate(correct)]
        (d / "stream.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    clean_case("no_defence", [True] * 20)
    clean_case("strict", [True] * 12 + [False] * 8)
    rows = {r["baseline"]: r for r in summarise_clean(load_clean_units(tmp_path))}
    assert rows["no_defence"]["clean_accuracy"] == 1.0
    assert rows["strict"]["clean_accuracy"] == 0.6 and rows["strict"]["clean_abstention"] == 0.4
    assert rows["strict"]["accuracy_loss_vs_reference"] == pytest.approx(0.4)
    assert rows["strict"]["p_exact"] < 0.01


def test_none_attack_ingests_no_poison_and_still_measures_accuracy(tmp_path):
    from trace_rag.evaluation.experiment import build_case_plan, load_fixture, run_matrix

    fixture = load_fixture(Path(__file__).resolve().parent.parent / "examples")
    plan = build_case_plan(fixture, "none", target_limit=3)
    assert plan.poison_doc_ids == [] and not any(e.is_poison for e in plan.ingestions)
    payload = run_matrix(["none"], ["no_defence"], [1], fixture, tmp_path / "o", target_limit=3,
                         scope_label="plumbing")
    assert payload["failures"] == []
    from trace_rag.evaluation.analysis import load_clean_units
    units = load_clean_units(tmp_path / "o")
    assert units and all(u["baseline"] == "no_defence" for u in units)
