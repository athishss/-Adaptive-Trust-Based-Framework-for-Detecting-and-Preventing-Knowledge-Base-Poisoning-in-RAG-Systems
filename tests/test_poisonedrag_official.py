"""Published PoisonedRAG poison: loading, injection and honest drop accounting.

The JSON below follows the schema of ``results/adv_targeted_results/<dataset>.json`` in the
authors' repository.  The *text* is a format fixture written for this test; real runs read
the authors' file.
"""
from __future__ import annotations

import json

import pytest

from trace_rag.evaluation.experiment import (ALL_ATTACKS, OFFICIAL_ATTACKS, SMOKE_ATTACKS,
                                             Fixture, build_case_plan, select_targets)
from trace_rag.evaluation.poisonedrag import load_poisonedrag_fixture

ADV = [f"Format fixture passage {i}: the capital of Freedonia is Sylvania." for i in range(5)]


def _write(tmp_path, records, gold=None, corpus_ids=None):
    poison = tmp_path / "poison.json"
    poison.write_text(json.dumps({r["id"]: r for r in records}), encoding="utf-8")
    corpus_ids = corpus_ids or ["d1", "d2", "d3", "d4", "d5", "d6"]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"_id": d, "title": f"T{d}", "text": f"text of {d}"})
                                for d in corpus_ids), encoding="utf-8")
    qrels = tmp_path / "qrels.tsv"
    gold = gold if gold is not None else {"q1": "d1", "q2": "d2"}
    qrels.write_text("query-id\tcorpus-id\tscore\n" + "\n".join(
        f"{q}\t{d}\t1" for q, d in gold.items()), encoding="utf-8")
    return poison, corpus, qrels


def _record(qid, **kw):
    base = {"id": qid, "question": f"question {qid}?", "correct answer": "Freeland",
            "incorrect answer": "Sylvania", "adv_texts": list(ADV)}
    base.update(kw)
    return base


def test_loader_keeps_gold_passages_and_reports_every_drop(tmp_path):
    records = [_record("q1"), _record("q2"), _record("q3"),               # q3: no qrels
               _record("q4"), _record("q5", **{"adv_texts": []})]          # q4: gold missing, q5 incomplete
    poison, corpus, qrels = _write(tmp_path, records,
                                   gold={"q1": "d1", "q2": "d2", "q4": "zzz"})
    fixture, report = load_poisonedrag_fixture(poison, corpus, qrels, target_limit=1)
    assert [q["qid"] for q in fixture.questions] == ["q1", "q2"]
    assert fixture.targets == ["q1"] and report["background_questions"] == 1
    assert report["dropped"] == {"incomplete_record": 1, "no_qrels": 1,
                                 "gold_passage_not_in_corpus": 1}
    assert fixture.false_answers == {"q1": "Sylvania", "q2": "Sylvania"}
    assert fixture.questions[0]["adv_texts"] == ADV            # verbatim, not regenerated


def test_max_clean_always_keeps_selected_gold_passages(tmp_path):
    poison, corpus, qrels = _write(tmp_path, [_record("q1"), _record("q2")])
    fixture, report = load_poisonedrag_fixture(poison, corpus, qrels, target_limit=1,
                                               max_clean=3, seed=1)
    ids = {row["_id"] for row in fixture.clean}
    assert {"d1", "d2"} <= ids and len(ids) == 3 and report["clean_passages"] == 3


def test_loader_requires_background_questions(tmp_path):
    poison, corpus, qrels = _write(tmp_path, [_record("q1"), _record("q2")])
    with pytest.raises(ValueError, match="background"):
        load_poisonedrag_fixture(poison, corpus, qrels, target_limit=2)


def test_loader_fails_loudly_when_no_gold_passage_is_present(tmp_path):
    poison, corpus, qrels = _write(tmp_path, [_record("q1")], gold={"q1": "nope"})
    with pytest.raises(ValueError, match="include-queries"):
        load_poisonedrag_fixture(poison, corpus, qrels)


def _fixture(tmp_path) -> Fixture:
    poison, corpus, qrels = _write(tmp_path, [_record("q1"), _record("q2"), _record("q3")],
                                   gold={"q1": "d1", "q2": "d2", "q3": "d3"})
    return load_poisonedrag_fixture(poison, corpus, qrels, target_limit=2)[0]


def test_official_attacks_are_not_in_the_default_smoke_matrix():
    assert set(OFFICIAL_ATTACKS) == {"poisonedrag_official", "poisonedrag_official_sybil"}
    assert not set(OFFICIAL_ATTACKS) & set(SMOKE_ATTACKS)
    assert set(OFFICIAL_ATTACKS) <= set(ALL_ATTACKS)


def test_single_source_variant_injects_published_text_verbatim(tmp_path):
    fixture = _fixture(tmp_path)
    plan = build_case_plan(fixture, "poisonedrag_official", target_limit=2)
    poison = [e for e in plan.ingestions if e.is_poison]
    assert len(poison) == 10 and {e.text for e in poison} == set(ADV)
    for target in range(2):
        sources = {e.source_id for e in poison if e.target_query == f"question q{target + 1}?"}
        assert len(sources) == 1
    assert all(e.family_id is None and e.attack_type == "poisonedrag_official" for e in poison)
    assert len(plan.poison_doc_ids) == 10 and len(set(plan.poison_doc_ids)) == 10
    assert plan.answer_ground_truth and plan.description["official_published_poison"] is True


def test_sybil_variant_uses_five_distinct_sources_per_question(tmp_path):
    plan = build_case_plan(_fixture(tmp_path), "poisonedrag_official_sybil", target_limit=2)
    poison = [e for e in plan.ingestions if e.is_poison]
    for q in ("question q1?", "question q2?"):
        assert len({e.source_id for e in poison if e.target_query == q}) == 5
    assert len({e.source_id for e in poison}) == 10          # no source shared across questions


def test_official_attack_refuses_targets_without_published_poison():
    fixture = Fixture(
        clean=[{"_id": "d1", "text": "alpha is first", "title": "A"}], late_clean=[],
        questions=[{"qid": "q1", "question": "what is first?", "gold_answer": "alpha",
                    "gold_answers": ["alpha"], "gold_doc_id": "d1"}],
        targets=["q1"], false_answers={"q1": "beta"})
    assert select_targets(fixture)[0].adv_texts == ()
    with pytest.raises(ValueError, match="adv_texts"):
        build_case_plan(fixture, "poisonedrag_official", target_limit=1)


def test_withheld_poison_is_reported_separately_from_flag_based_detection(tmp_path):
    """RAGLedger kept the poison out of the context without flagging it: that must be visible."""
    from trace_rag.evaluation.experiment import run_matrix
    from trace_rag.evaluation.poisonedrag import load_poisonedrag_fixture
    from trace_rag.evaluation.experiment import load_fixture
    from pathlib import Path

    base = load_fixture(Path(__file__).resolve().parent.parent / "examples")
    questions = base.questions[:5]
    poison = {}
    gold = []
    for i, q in enumerate(questions):
        poison[f"q{i}"] = {"id": f"q{i}", "question": q["question"],
                           "correct answer": q.get("gold_answer", "x"),
                           "incorrect answer": "Zog the Alien",
                           "adv_texts": [f"{q['question']} The answer is Zog the Alien ({j})."
                                         for j in range(5)]}
        answer = q.get("gold_answer", "")
        gold.append(f"q{i}\t" + next(r["_id"] for r in base.clean if answer.lower() in r["text"].lower()) + "\t1")
    (tmp_path / "p.json").write_text(json.dumps(poison))
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(
        {"_id": r["_id"], "title": r.get("title", ""), "text": r["text"]}) for r in base.clean))
    (tmp_path / "q.tsv").write_text("query-id\tcorpus-id\tscore\n" + "\n".join(gold))
    fixture, _ = load_poisonedrag_fixture(tmp_path / "p.json", tmp_path / "c.jsonl",
                                          tmp_path / "q.tsv", target_limit=2)
    payload = run_matrix(["poisonedrag_official"], ["no_defence", "RAGLedger"], [1], fixture,
                         tmp_path / "out", target_limit=2, repeat_targets=2,
                         scope_label="plumbing")
    assert payload["failures"] == []
    rows = {r["case"]["baseline"]: r for r in payload["results"]}
    undefended, ledger = rows["no_defence"], rows["RAGLedger"]
    assert undefended["withheld_poison_recall"] == 0.0
    assert undefended["recall_defined"] is True and undefended["recall"] == 0.0
    for key in ("withheld_poison_recall", "withheld_clean_rate", "precision_defined"):
        assert key in ledger
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "withheld poison" in md and "n/a means" in md
