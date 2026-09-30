from __future__ import annotations

import pytest

from trace_rag.contracts import AnswerRecord, Citation
from trace_rag.provenance import AnswerLog, RemediationService


def record(query_id: str, cited, abstained: bool = False, trust: float = 0.7) -> AnswerRecord:
    return AnswerRecord(
        query_id=query_id, query=f"question {query_id}", answer="answer text",
        abstained=abstained, abstain_reason=None,
        citations=tuple(Citation(c, c, "sentence") for c in cited),
        used_doc_ids=tuple(cited), excluded_doc_ids=(),
        trust_snapshots={c: {"t_eff": trust} for c in cited},
        evidence_mass=trust, llm_calls=1, latency_ms=12.0,
    )


@pytest.fixture
def log():
    with AnswerLog(":memory:") as answer_log:
        yield answer_log


def test_answers_are_recorded_with_their_documents(log):
    answer_id = log.record(record("q1", ["d1", "d2"]), context_doc_ids=["d1", "d2", "d3"])
    stored = log.get(answer_id)
    assert stored.cited_doc_ids == ("d1", "d2")
    assert log.payload(answer_id)["query_id"] == "q1"


def test_answers_using_covers_cited_and_context(log):
    log.record(record("q1", ["d1"]), context_doc_ids=["d1", "d9"])
    log.record(record("q2", ["d2"]), context_doc_ids=["d2"])
    assert {a.query_id for a in log.answers_using(["d9"])} == {"q1"}
    assert {a.query_id for a in log.answers_using(["d2"])} == {"q2"}
    assert log.answers_using(["missing"]) == []


def test_remediation_flags_only_affected_answers(log):
    a1 = log.record(record("q1", ["d1", "d2"]))
    a2 = log.record(record("q2", ["d3"]))
    a3 = log.record(record("q3", ["d1"]))
    report = RemediationService(log).on_quarantine(["d1"], reason="refuted twice")
    assert set(report.affected_answer_ids) == {a1, a3}
    assert report.newly_flagged == 2 and report.exposure_window == 2
    assert log.get(a2).flagged is False
    assert log.get(a1).flag_reason == "refuted twice"


def test_remediation_is_idempotent(log):
    log.record(record("q1", ["d1"]))
    service = RemediationService(log)
    assert service.on_quarantine(["d1"]).newly_flagged == 1
    second = service.on_quarantine(["d1"])
    assert second.newly_flagged == 0 and second.already_flagged == 1


def test_dry_run_reports_without_writing(log):
    answer_id = log.record(record("q1", ["d1"]))
    report = RemediationService(log).on_quarantine(["d1"], dry_run=True)
    assert report.affected_answer_ids == (answer_id,) and report.newly_flagged == 0
    assert log.get(answer_id).flagged is False


def test_exposure_window_ignores_abstentions(log):
    log.record(record("q1", ["d1"], abstained=True))
    log.record(record("q2", ["d1"]))
    assert RemediationService(log).on_quarantine(["d1"]).exposure_window == 1


def test_remediation_recall_uses_ground_truth_ids(log):
    a1 = log.record(record("q1", ["d1"]))
    a2 = log.record(record("q2", ["d2"]))
    service = RemediationService(log)
    service.on_quarantine(["d1"])
    assert service.remediation_recall([a1, a2]) == 0.5
    assert service.remediation_recall([]) is None


def test_stats_summarise_the_log(log):
    log.record(record("q1", ["d1"]))
    log.record(record("q2", [], abstained=True))
    stats = log.stats()
    assert stats["answers"] == 2 and stats["abstained"] == 1
    assert stats["abstention_rate"] == 0.5 and stats["mean_llm_calls"] == 1.0


def test_log_survives_reopen(tmp_path):
    path = tmp_path / "answers.sqlite3"
    with AnswerLog(path) as log:
        log.record(record("q1", ["d1"]))
    with AnswerLog(path) as reopened:
        assert reopened.stats()["answers"] == 1
        assert RemediationService(reopened).on_quarantine(["d1"]).newly_flagged == 1
