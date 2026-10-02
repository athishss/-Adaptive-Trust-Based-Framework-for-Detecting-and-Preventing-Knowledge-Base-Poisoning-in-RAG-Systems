"""End-to-end tests, including the seams Person B and Person C plug into."""

from __future__ import annotations

import json


from trace_rag.contracts import (Band, PolicyDecision, TrustSnapshot, TrustStatus,
                                 VerificationOutcome, VerificationResult)
from trace_rag.pipeline import PersonAPipeline

QUESTION = "Who designed the Eiffel Tower?"


class FakeLedger:
    """Stand-in for Person B's trust ledger, with the same protocol."""

    def __init__(self) -> None:
        self.trust = {}
        self.blocked = set()
        self.calls = 0

    def set(self, doc_id: str, value: float, status: TrustStatus = TrustStatus.MONITORED) -> None:
        self.trust[doc_id] = (value, status)
        if status in (TrustStatus.QUARANTINED, TrustStatus.REJECTED):
            self.blocked.add(doc_id)

    def get_trust(self, doc_ids):
        self.calls += 1
        out = {}
        for doc_id in doc_ids:
            value, status = self.trust.get(doc_id, (0.5, TrustStatus.TRUSTED))
            out[doc_id] = TrustSnapshot(doc_id, "s", "f", value, value, value, value, status)
        return out

    def blocked_doc_ids(self):
        return set(self.blocked)


class FakeVerifier:
    """Refutes any passage that echoes the question; counts its LLM calls."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, query, query_id, target, pool):
        self.calls += 1
        refute = query.lower().rstrip("?") in target.text.lower()
        return VerificationResult(
            doc_id=target.doc_id, query_id=query_id, single_doc_answer="claim",
            influential=True, support_mass=0.0 if refute else 0.8,
            refute_mass=0.9 if refute else 0.0,
            outcome=VerificationOutcome.REFUTE if refute else VerificationOutcome.SUPPORT,
            llm_calls=2, latency_ms=5.0,
        )


def test_pipeline_answers_and_logs(populated_pipeline):
    result = populated_pipeline.answer(QUESTION, "q1")
    assert result.answer_id is not None
    assert result.assessments and result.record.query_id == "q1"
    assert populated_pipeline.answer_log.get(result.answer_id).query_id == "q1"
    assert set(result.timings_ms) >= {"retrieval", "signals_and_scoring", "policy", "generation"}


def test_result_is_json_serialisable(populated_pipeline):
    json.dumps(populated_pipeline.answer(QUESTION, "q1").to_dict())


def test_pipeline_runs_without_person_b(populated_pipeline):
    result = populated_pipeline.answer(QUESTION, "q1")
    assert type(populated_pipeline.trust_provider).__name__ == "NullTrustProvider"
    assert result.llm_calls >= 1


def test_person_b_trust_provider_is_used(populated_pipeline):
    ledger = FakeLedger()
    populated_pipeline.retriever.trust = ledger
    populated_pipeline.answer(QUESTION, "q1")
    assert ledger.calls > 0


def test_quarantined_passages_never_reach_the_answer(populated_pipeline):
    ledger = FakeLedger()
    for doc_id in ("poison_1#0000", "poison_2#0000", "poison_3#0000"):
        ledger.set(doc_id, 0.05, TrustStatus.QUARANTINED)
    populated_pipeline.retriever.trust = ledger
    result = populated_pipeline.answer(QUESTION, "q1")
    assert all(not d.doc_id.startswith("poison") for d in result.retrieval.documents)
    assert all(not c.doc_id.startswith("poison") for c in result.record.citations)


def test_verifier_is_called_only_for_medium_band(populated_pipeline):
    verifier = FakeVerifier()
    populated_pipeline.verifier = verifier
    result = populated_pipeline.answer(QUESTION, "q1")
    medium = [a for a in result.assessments if a.band is Band.MEDIUM]
    assert verifier.calls == len(medium)
    assert result.llm_calls == result.record.llm_calls + 2 * verifier.calls


def test_refuted_passages_are_excluded_from_context(populated_pipeline):
    populated_pipeline.verifier = FakeVerifier()
    populated_pipeline.scorer.theta_low = 0.0          # force everything into MEDIUM
    populated_pipeline.scorer.theta_high = 1.01
    result = populated_pipeline.answer(QUESTION, "q1")
    refuted = {v.doc_id for v in result.decision.verified
               if v.outcome is VerificationOutcome.REFUTE}
    assert refuted
    assert refuted.isdisjoint(set(result.decision.context_doc_ids))
    assert refuted.issubset(set(result.decision.excluded_doc_ids))


def test_custom_policy_overrides_default(populated_pipeline):
    class ExcludeEverything:
        def decide(self, query, query_id, documents, assessments, verifier=None, **kwargs):
            return PolicyDecision(context_doc_ids=(),
                                  excluded_doc_ids=tuple(d.doc_id for d in documents))

    populated_pipeline.policy = ExcludeEverything()
    result = populated_pipeline.answer(QUESTION, "q1")
    assert result.abstained and result.record.abstain_reason == "no_context"


def test_quarantine_triggers_retroactive_remediation(populated_pipeline):
    first = populated_pipeline.answer(QUESTION, "q1")
    cited = list(first.record.used_doc_ids)
    assert cited, "test needs a cited passage"
    report = populated_pipeline.on_quarantine(cited[:1], reason="refuted by verifier")
    assert first.answer_id in report.affected_answer_ids
    assert populated_pipeline.answer_log.get(first.answer_id).flagged is True


def test_answer_stream_preserves_step_numbers(populated_pipeline):
    stream = [{"t": 10, "qid": "s1", "text": QUESTION},
              {"t": 11, "qid": "s2", "text": "What is photosynthesis?"}]
    results = populated_pipeline.answer_stream(stream)
    assert [r.record.step for r in results] == [10, 11]
    assert [r.query_id for r in results] == ["s1", "s2"]


def test_pipeline_is_deterministic(populated_pipeline):
    a = populated_pipeline.answer(QUESTION, "q1")
    b = populated_pipeline.answer(QUESTION, "q2")
    assert [d.doc_id for d in a.retrieval.documents] == [d.doc_id for d in b.retrieval.documents]
    assert [round(x.suspicion, 9) for x in a.assessments] == [round(x.suspicion, 9) for x in b.assessments]
    assert a.record.answer == b.record.answer


def test_stats_report_wiring(populated_pipeline):
    stats = populated_pipeline.stats()
    assert stats["index_size"] == 11 and stats["store"]["chunks"] == 11
    assert stats["policy"] == "DefaultPolicy"


def test_incremental_indexing_of_new_passages(populated_pipeline):
    from trace_rag.ingestion import Ingestor

    ingestor = Ingestor(populated_pipeline.store, populated_pipeline.config.ingestion)
    records = ingestor.ingest_text("late_1", "A newly injected passage about the Eiffel Tower.",
                                   "attacker_2", ingested_at=1_800_000_000.0, passage_mode=True)
    added = populated_pipeline.index_chunks([r.chunk_id for r in records])
    assert added == len(records)
    assert "late_1#0000" in populated_pipeline.index


def test_index_persists_across_pipeline_instances(config):
    from trace_rag.ingestion import Ingestor

    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    Ingestor(pipeline.store, config.ingestion).ingest_text(
        "d1", "Gustave Eiffel designed the tower.", "wiki", ingested_at=1.0, passage_mode=True)
    pipeline.index_chunks()
    pipeline.save_index()
    pipeline.close()

    reopened = PersonAPipeline.from_config(config, load_existing_index=True)
    assert len(reopened.index) == 1
    assert reopened.answer("Who designed the tower?", "q1").retrieval.documents
    reopened.close()
