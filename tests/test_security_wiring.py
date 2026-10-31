"""The protected configuration must be the default, and it must say so.

These tests close the gap behind the original integration defect: the
documented pipeline entry point built a null trust provider and ``DefaultPolicy``
unless a caller wired Person B by hand, so ``trace-rag query`` was an
unprotected RAG system with a security module sitting unused next to it.
"""

from __future__ import annotations

from trace_rag.contracts import (
    Action,
    Band,
    DefaultPolicy,
    FeatureSnapshot,
    NullTrustProvider,
    RetrievedDocument,
    SecurityAssessment,
    SignalVector,
    TrustSnapshot,
)
import numpy as np

from trace_rag import security as security_module
from trace_rag.pipeline import PersonAPipeline
from trace_rag.security import DEGRADED, PROTECTED, UNPROTECTED, security_health
from trace_rag.trust.ledger import TrustLedger
from trace_rag.trust.policy import TrustPolicy
from trace_rag.trust.verifier import CorroborationVerifier


def test_default_config_enables_the_security_stack(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        health = pipeline.health_check()
        assert isinstance(pipeline.trust_provider, TrustLedger)
        assert isinstance(pipeline.verifier, CorroborationVerifier)
        assert isinstance(pipeline.policy, TrustPolicy)
        assert pipeline.security is not None
        assert health["security_enabled"] is True
        assert health["queue_enabled"] is True
        assert health["trust_persistent"] is True
        # The offline backends make this DEGRADED, never PROTECTED.  Reporting
        # DEGRADED is the point: the controls are real, the models are not.
        assert health["status"] == DEGRADED
        assert health["research_ready"] is False
        assert health["smoke_test_backends"] is True
        assert any("embedding" in item for item in health["fallbacks"])
        assert any("generation" in item for item in health["fallbacks"])
        assert health["readiness_probes"]["embedding"]["status"] == "fallback"
        assert health["readiness_probes"]["generation"]["status"] == "fallback"
        assert health["readiness_ready"] is False
        assert "no model downloads" in health["probe_policy"]
    finally:
        pipeline.close()


def test_explicit_unprotected_baseline_is_reported_as_unprotected(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False,
                                           secure=False)
    try:
        health = pipeline.health_check()
        assert isinstance(pipeline.trust_provider, NullTrustProvider)
        assert isinstance(pipeline.policy, DefaultPolicy)
        assert health["status"] == UNPROTECTED
        assert health["security_enabled"] is False
    finally:
        pipeline.close()


class _ReadyEmbedder:
    name = "huggingface"
    model_name = "fake/real-embedder"
    dim = 4

    def encode_queries(self, texts, batch_size=32):
        return np.ones((len(texts), self.dim), dtype=np.float32)


class _BrokenEmbedder(_ReadyEmbedder):
    def encode_queries(self, texts, batch_size=32):
        raise RuntimeError("CUDA model not loaded")


class _ReadyLLM:
    name = "fake/real-llm"
    base_url = "http://localhost:9999/v1"

    def generate(self, prompt, max_tokens=256, stop=None):
        raise AssertionError("readiness must not generate text")


class _FakeNLI:
    def __init__(self, ready=True):
        self.mode = "nli" if ready else "lexical"
        self.nli_ready = ready
        self.nli_error = None if ready else "model download failed"


def _ready_pipeline(pipeline, monkeypatch, embedder=None):
    monkeypatch.setattr(security_module, "_http_json",
                        lambda *args, **kwargs: {"data": [{"id": "fake/real-llm"}]})
    pipeline.embedder = embedder or _ReadyEmbedder()
    pipeline.generator.llm = _ReadyLLM()
    pipeline.verifier = _FakeNLI(ready=True)
    pipeline.config.security.verifier = "nli"
    return pipeline


def test_protected_requires_working_probes_not_configured_names(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        health = security_health(pipeline)
        assert health["status"] == PROTECTED
        assert health["readiness_ready"] is True
        assert health["readiness_probes"]["embedding"]["status"] == "ready"
        assert health["readiness_probes"]["generation"]["status"] == "ready"
        assert health["readiness_probes"]["nli_verifier"]["status"] == "ready"

        # Same configured model names, but the embedding backend fails its
        # probe: PROTECTED is withdrawn and the error is explicit.
        pipeline.embedder = _BrokenEmbedder()
        degraded = security_health(pipeline)
        assert degraded["status"] == DEGRADED
        assert any("embedding" in item for item in degraded["readiness_failures"])
        assert "CUDA model not loaded" in degraded["readiness_probes"]["embedding"]["detail"]
    finally:
        pipeline.close()


def test_requested_nli_that_failed_to_load_is_not_protected(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        pipeline.verifier = _FakeNLI(ready=False)
        pipeline.config.security.verifier = "nli"
        health = security_health(pipeline)
        assert health["status"] == DEGRADED
        assert health["readiness_probes"]["nli_verifier"]["status"] == "failed"
        assert any("nli_verifier" in item for item in health["readiness_failures"])
    finally:
        pipeline.close()


def test_medium_without_a_verifier_fails_closed(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        doc = RetrievedDocument(
            doc_id="d1#0000", chunk_id="d1#0000", text="a suspicious passage",
            similarity=0.8, rank=0, source_id="s1", family_id="f1",
            trust=TrustSnapshot(doc_id="d1#0000", source_id="s1", family_id="f1"),
        )
        assessment = SecurityAssessment(
            doc_id=doc.doc_id, chunk_id=doc.chunk_id, query_id="q1",
            signals=SignalVector(), suspicion=0.5, band=Band.MEDIUM,
            action=Action.VERIFY_THEN_USE,
            feature_snapshot=FeatureSnapshot(signals=SignalVector(), extras={}),
        )
        policy = TrustPolicy(ledger=pipeline.trust_provider,
                             fail_open_without_verifier=False)
        decision = policy.decide(
            "who designed the tower?", "q1", [doc], [assessment], verifier=None,
        )
        assert decision.context_doc_ids == ()
        assert decision.excluded_doc_ids == (doc.doc_id,)
        assert decision.notes["decisions"][doc.doc_id]["reason"] == \
            "medium_no_verifier_fail_closed"
    finally:
        pipeline.close()
