"""End-to-end smoke tests for the chronological harness.

These run the real ingestion -> retrieval -> signals -> policy -> verification
-> trust -> generation -> remediation path on the bundled mini corpus, using the
offline backends.  They are deliberately small and deterministic; they are not
benchmark measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_rag.attacks.stream import (
    StreamContractError,
    as_ingestion_row,
    as_query_row,
    zipf_queries,
)
from trace_rag.evaluation.experiment import (
    EvaluationCase,
    execute_case,
    load_fixture,
    run_matrix,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SEED = 20260921


def test_stream_adapters_accept_legacy_and_canonical_rows():
    assert as_query_row({"t": 3, "qid": "q3", "text": "Who?"}) == {
        "step": 3, "query_id": "q3", "query": "Who?",
    }
    assert as_query_row({"step": 4, "query_id": "q4", "query": "Why?"}) == {
        "step": 4, "query_id": "q4", "query": "Why?",
    }
    ingestion = as_ingestion_row({
        "t": 1, "id": "d1", "payload": "A passage.", "source_id": "s1",
    })
    assert ingestion["step"] == 1
    assert ingestion["doc_id"] == "d1"
    assert ingestion["text"] == "A passage."

    with pytest.raises(StreamContractError):
        as_query_row({"qid": "missing-query"})
    with pytest.raises(StreamContractError):
        as_ingestion_row({"step": 0, "doc_id": "d1", "text": " ", "source_id": "s"})


def test_zipf_queries_is_deterministic_and_returns_known_questions():
    questions = ["alpha?", "beta?", "gamma?"]
    first = zipf_queries(questions, steps=40, seed=7)
    second = zipf_queries(questions, steps=40, seed=7)
    assert len(first) == 40
    assert first == second
    assert set(first) <= set(questions)
    assert zipf_queries(questions, steps=0) == []
    with pytest.raises(ValueError):
        zipf_queries([], steps=1)


def test_entity_swap_runs_the_detection_quarantine_lifecycle(tmp_path):
    """Poison is refuted twice, quarantined, and blocked from later answers."""
    fixture = load_fixture(EXAMPLES)
    case = EvaluationCase(attack="entity_swap", baseline="RAGLedger", seed=SEED)
    # Four repetitions: repeated refutations quarantine one passage, and the
    # remaining repetitions are genuine post-quarantine target queries that
    # must no longer retrieve it.  Which target is quarantined depends on the
    # retrieval schedule, so the test follows the reported quarantine instead
    # of hard-coding one fixture id.
    result = execute_case(case, fixture, tmp_path, target_limit=6, repeat_targets=4)

    metrics = result.details["metrics"]
    # Six targets asked four times each, minus the steps where consecutive
    # targets overlap.  Attempts stop being counted once the poison is no
    # longer retrieved, so both counts are lower bounds.
    assert metrics["target_queries"] >= 18
    assert metrics["eligible_attack_attempts"] >= 18
    assert metrics["poison_exposure_rate"] == 0.0
    assert metrics["poison_citation_rate"] == 0.0
    assert metrics["answer_level_asr"] is None
    assert metrics["answer_level_asr_status"] == "NOT MEASURED"
    assert metrics["poison_exposed_attempts"] == 0
    assert metrics["poison_quarantined"], "the entity swap must be quarantined"
    assert metrics["clean_quarantined"] == []
    assert metrics["detection"]["precision"] == 1.0
    assert metrics["detection"]["recall"] > 0.0
    assert metrics["verification"]["REFUTE"] > 0

    stream_path = Path(result.details["case_dir"]) / "stream.jsonl"
    records = [json.loads(line) for line in
               stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    quarantined = list(metrics["poison_quarantined"])
    assert quarantined, "at least one entity-swap passage must be quarantined"
    quarantine_steps = {
        doc_id: record["step"]
        for record in records
        for doc_id in record["newly_quarantined"]
    }
    quarantined_doc = next(doc_id for doc_id in quarantined
                           if doc_id in quarantine_steps)
    quarantined_at = quarantine_steps[quarantined_doc]

    later = [record for record in records
             if record["step"] > quarantined_at and record["is_target"]
             and quarantined_doc in record["poison_targeted"]]
    assert later, "a later target query must exercise the exclusion path"
    for record in later:
        assert quarantined_doc not in record["context_ids"]
        assert quarantined_doc not in record["poison_in_context"]


def test_run_matrix_writes_machine_readable_results(tmp_path):
    payload = run_matrix(
        attacks=["negation"], baselines=["no_defence"], seeds=[1],
        fixture=load_fixture(EXAMPLES), output_root=tmp_path / "matrix",
        target_limit=1, repeat_targets=2,
    )

    assert payload["cases_completed"] == 1
    assert payload["failures"] == []
    out = tmp_path / "matrix"
    assert (out / "results.json").exists()
    assert (out / "results.csv").exists()
    assert (out / "summary.md").exists()
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["scope"].startswith("SMOKE TEST")
    assert data["results"][0]["case"]["baseline"] == "no_defence"
    assert data["results"][0]["answer_level_asr_status"] == "NOT MEASURED"
    assert "attack_success_rate" not in data["results"][0]
    assert (out / "summary.md").read_text(encoding="utf-8").count("NOT MEASURED") >= 1
