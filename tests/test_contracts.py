"""The contract is the integration surface: these tests are the ones Person B and C rely on."""

from __future__ import annotations

import json

import pytest

from trace_rag.contracts import (Action, AnswerRecord, Band, Citation, DefaultPolicy,
                                 FeatureSnapshot, LLM, NullTrustProvider, NullVerifier,
                                 Policy, RetrievedDocument, SecurityAssessment, SignalVector,
                                 TrustProvider, TrustSnapshot, TrustStatus, Verifier,
                                 VerificationOutcome, VerificationResult, FEATURE_NAMES,
                                 SIGNAL_NAMES)
from conftest import make_doc


def test_signal_vector_array_order_matches_names():
    signals = SignalVector(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert list(signals.as_array()) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert tuple(signals.to_dict()) == SIGNAL_NAMES


def test_feature_snapshot_respects_feature_order():
    snapshot = FeatureSnapshot(signals=SignalVector(1, 0, 0, 0, 0, 0),
                               extras={"x_similarity": 0.9, "x_rank": 3.0})
    array = snapshot.as_array()
    assert array.shape == (len(FEATURE_NAMES),)
    assert array[0] == 1.0
    assert array[FEATURE_NAMES.index("x_similarity")] == 0.9
    assert array[FEATURE_NAMES.index("x_source_trust")] == 0.0   # missing extras default to 0


def test_all_payloads_are_json_serialisable():
    doc = make_doc("d1")
    assessment = SecurityAssessment(
        doc_id="d1", chunk_id="d1", query_id="q1", signals=SignalVector(),
        suspicion=0.4, band=Band.MEDIUM, action=Action.VERIFY_THEN_USE,
        feature_snapshot=FeatureSnapshot(SignalVector(), {"x_similarity": 0.5}),
    )
    record = AnswerRecord(query_id="q1", query="q", answer="a", abstained=False,
                          abstain_reason=None, citations=(Citation("d1", "d1", "s"),),
                          used_doc_ids=("d1",), excluded_doc_ids=(), trust_snapshots={},
                          evidence_mass=0.7, llm_calls=1, latency_ms=2.0)
    result = VerificationResult("d1", "q1", "answer", True, 0.8, 0.1,
                                VerificationOutcome.SUPPORT, 2, 5.0)
    for payload in (doc.to_dict(), assessment.to_dict(), record.to_dict(), result.to_dict()):
        json.dumps(payload)                      # must not raise


def test_enums_serialise_as_plain_strings():
    assert json.loads(json.dumps({"b": Band.HIGH.value}))["b"] == "HIGH"
    assert TrustStatus.QUARANTINED.value == "QUARANTINED"


def test_null_implementations_satisfy_protocols():
    assert isinstance(NullTrustProvider(), TrustProvider)
    assert isinstance(NullVerifier(), Verifier)
    assert isinstance(DefaultPolicy(), Policy)


def test_null_trust_provider_is_neutral():
    provider = NullTrustProvider()
    trust = provider.get_trust(["a", "b"])
    assert set(trust) == {"a", "b"}
    assert trust["a"].t_eff == 0.5
    assert provider.blocked_doc_ids() == set()


def test_default_policy_bands_drive_context():
    docs = [make_doc("low"), make_doc("mid"), make_doc("high")]
    assessments = [
        SecurityAssessment("low", "low", "q", SignalVector(), 0.1, Band.LOW, Action.USE,
                           FeatureSnapshot(SignalVector(), {})),
        SecurityAssessment("mid", "mid", "q", SignalVector(), 0.5, Band.MEDIUM,
                           Action.VERIFY_THEN_USE, FeatureSnapshot(SignalVector(), {})),
        SecurityAssessment("high", "high", "q", SignalVector(), 0.9, Band.HIGH, Action.EXCLUDE,
                           FeatureSnapshot(SignalVector(), {})),
    ]
    decision = DefaultPolicy().decide("q", "q1", docs, assessments, verifier=None)
    assert decision.context_doc_ids == ("low", "mid")
    assert decision.excluded_doc_ids == ("high",)


def test_default_policy_drops_refuted_documents():
    class RefutingVerifier:
        def verify(self, query, query_id, target, pool):
            return VerificationResult(target.doc_id, query_id, "x", True, 0.0, 0.9,
                                       VerificationOutcome.REFUTE, 2, 1.0)

    docs = [make_doc("mid")]
    assessments = [SecurityAssessment("mid", "mid", "q", SignalVector(), 0.5, Band.MEDIUM,
                                      Action.VERIFY_THEN_USE, FeatureSnapshot(SignalVector(), {}))]
    decision = DefaultPolicy().decide("q", "q1", docs, assessments, RefutingVerifier())
    assert decision.context_doc_ids == ()
    assert decision.excluded_doc_ids == ("mid",)
    assert decision.verified[0].outcome is VerificationOutcome.REFUTE


def test_trust_snapshot_neutral_defaults():
    snapshot = TrustSnapshot.neutral("d1")
    assert snapshot.t_eff == 0.5 and snapshot.status is TrustStatus.TRUSTED
    assert snapshot.to_dict()["status"] == "TRUSTED"
