"""The protected configuration must be the default, and it must say so.

These tests close the gap behind the original integration defect: the
documented pipeline entry point built a null trust provider and ``DefaultPolicy``
unless a caller wired Person B by hand, so ``trace-rag query`` was an
unprotected RAG system with a security module sitting unused next to it.
"""

from __future__ import annotations

import json

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
from trace_rag.generation.llm import LLMError, LLMResponse
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
        return LLMResponse(text="OK", llm_calls=1, model=self.name)


class _HFLLM(_ReadyLLM):
    name = "fake/hf-llm"
    base_url = ""
    tokenizer = object()
    model = object()
    device = "cpu"

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, max_tokens=256, stop=None):
        return LLMResponse(text=self.text, llm_calls=1, model=self.name)


class _NaNEmbedder(_ReadyEmbedder):
    def encode_queries(self, texts, batch_size=32):
        return np.full((len(texts), self.dim), np.nan, dtype=np.float32)


class _ZeroEmbedder(_ReadyEmbedder):
    def encode_queries(self, texts, batch_size=32):
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class _WrongDimEmbedder(_ReadyEmbedder):
    def encode_queries(self, texts, batch_size=32):
        return np.ones((len(texts), self.dim + 1), dtype=np.float32)


class _FakeNLI:
    def __init__(self, ready=True):
        self.mode = "nli" if ready else "lexical"
        self.nli_ready = ready
        self.nli_error = None if ready else "model download failed"

    def nli_probe(self):
        if not self.nli_ready:
            return {"status": "failed", "detail": self.nli_error}
        return {"status": "ready", "label": "entailment", "score": 0.99,
                "model": "fake-nli"}


class _BrokenProbeNLI(_FakeNLI):
    def __init__(self):
        super().__init__(ready=True)

    def nli_probe(self):
        raise RuntimeError("NLI inference crashed")


def _ready_pipeline(pipeline, monkeypatch, embedder=None):
    monkeypatch.setattr(security_module, "_http_json",
                        lambda *args, **kwargs: {"data": [{"id": "fake/real-llm"}]})
    monkeypatch.setattr(security_module, "_http_post_json",
                        lambda *args, **kwargs: {
                            "model": "fake/real-llm",
                            "choices": [{"message": {"content": "OK"}}],
                        })
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
        assert health["readiness_probes"]["generation"]["served_model"] == "fake/real-llm"
        assert health["readiness_probes"]["nli_verifier"]["label"] == "entailment"
        assert health["research_ready"] is True

        # Same configured model names, but the embedding backend fails its
        # probe: PROTECTED is withdrawn and the error is explicit.
        pipeline.embedder = _BrokenEmbedder()
        degraded = security_health(pipeline)
        assert degraded["status"] == DEGRADED
        assert any("embedding" in item for item in degraded["readiness_failures"])
        assert "CUDA model not loaded" in degraded["readiness_probes"]["embedding"]["detail"]
    finally:
        pipeline.close()


def test_embedding_probe_checks_finite_values_and_dimensions(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch, embedder=_NaNEmbedder())
        health = security_health(pipeline)
        assert health["readiness_probes"]["embedding"]["status"] == "failed"
        assert "non-finite" in health["readiness_probes"]["embedding"]["detail"]
        assert health["status"] == DEGRADED

        _ready_pipeline(pipeline, monkeypatch, embedder=_ZeroEmbedder())
        health = security_health(pipeline)
        assert health["readiness_probes"]["embedding"]["status"] == "failed"
        assert "all-zero" in health["readiness_probes"]["embedding"]["detail"]

        _ready_pipeline(pipeline, monkeypatch, embedder=_WrongDimEmbedder())
        health = security_health(pipeline)
        assert health["readiness_probes"]["embedding"]["status"] == "failed"
        assert "expected 4" in health["readiness_probes"]["embedding"]["detail"]
    finally:
        pipeline.close()


def test_generator_metadata_alone_is_not_readiness(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        monkeypatch.setattr(security_module, "_http_post_json",
                            lambda *args, **kwargs: (_ for _ in ()).throw(
                                LLMError("model crashed during completion")))
        health = security_health(pipeline)
        assert health["readiness_probes"]["generation"]["status"] == "failed"
        assert "model crashed" in health["readiness_probes"]["generation"]["detail"]
        assert health["status"] == DEGRADED
        assert health["research_ready"] is False
    finally:
        pipeline.close()


def test_generator_probe_accepts_a_valid_minimal_response(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        probe = security_health(pipeline)["readiness_probes"]["generation"]
        assert probe["status"] == "ready"
        assert probe["probe_response_chars"] >= 1
        assert probe["served_model"] == "fake/real-llm"
    finally:
        pipeline.close()


def test_huggingface_generation_rejects_empty_and_whitespace(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        pipeline.config.generation.backend = "huggingface"
        for text in ("", "   ", "\n\t"):
            pipeline.generator.llm = _HFLLM(text)
            probe = security_health(pipeline)["readiness_probes"]["generation"]
            assert probe["status"] == "failed"
            assert "empty" in probe["detail"]
    finally:
        pipeline.close()


def test_openai_probe_rejects_missing_or_empty_assistant_text(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        cases = [
            {},
            {"choices": []},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": "   "}}]},
        ]
        for body in cases:
            monkeypatch.setattr(security_module, "_http_post_json",
                                lambda *args, _body=body, **kwargs: _body)
            probe = security_health(pipeline)["readiness_probes"]["generation"]
            assert probe["status"] == "failed"
            assert security_health(pipeline)["status"] != PROTECTED
    finally:
        pipeline.close()


def test_ollama_probe_rejects_empty_generated_response(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        pipeline.config.generation.backend = "ollama"
        pipeline.generator.llm.base_url = "http://localhost:11434"
        monkeypatch.setattr(security_module, "_http_json",
                            lambda *args, **kwargs: {"models": [{"name": "fake/real-llm"}]})
        monkeypatch.setattr(security_module, "_http_post_json",
                            lambda *args, **kwargs: {"response": "  "})
        probe = security_health(pipeline)["readiness_probes"]["generation"]
        assert probe["status"] == "failed"
        assert "empty" in probe["detail"]
    finally:
        pipeline.close()


def test_http_timeout_is_a_failed_probe_not_protected(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        monkeypatch.setattr(security_module, "_http_json",
                            lambda *args, **kwargs: (_ for _ in ()).throw(
                                TimeoutError("timed out")))
        health = security_health(pipeline)
        probe = health["readiness_probes"]["generation"]
        assert probe["status"] == "failed"
        assert "endpoint unavailable" in probe["detail"]
        assert health["status"] == DEGRADED
        assert health["research_ready"] is False
    finally:
        pipeline.close()


def test_api_keys_are_redacted_from_readiness_reports(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        secret = "sk-super-secret-value"
        pipeline.generator.llm.api_key = secret
        monkeypatch.setattr(security_module, "_http_post_json",
                            lambda *args, **kwargs: (_ for _ in ()).throw(
                                RuntimeError(f"Authorization: Bearer {secret} rejected")))
        health = security_health(pipeline)
        assert health["readiness_probes"]["generation"]["status"] == "failed"
        assert secret not in json.dumps(health)
        assert "[REDACTED]" in health["readiness_probes"]["generation"]["detail"]
    finally:
        pipeline.close()


def test_nli_loaded_but_inference_fails_is_not_protected(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        pipeline.verifier = _BrokenProbeNLI()
        health = security_health(pipeline)
        assert health["readiness_probes"]["nli_verifier"]["status"] == "failed"
        assert "inference crashed" in health["readiness_probes"]["nli_verifier"]["detail"]
        assert health["status"] == DEGRADED
    finally:
        pipeline.close()


def test_nli_probe_validates_scores_and_contract_labels(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        probe = security_health(pipeline)["readiness_probes"]["nli_verifier"]
        assert probe["status"] == "ready"
        assert probe["label"] in {"entailment", "contradiction", "neutral"}
        assert 0.0 <= probe["score"] <= 1.0
    finally:
        pipeline.close()


def test_malformed_nli_outputs_cannot_be_protected(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        cases = [
            ({"status": "ready", "label": "entailment"}, "non-numeric NLI score"),
            ({"status": "ready", "score": 0.99}, "unexpected NLI label"),
            ({"status": "ready", "label": "entailment", "score": float("nan")},
             "invalid NLI score"),
            ({"status": "ready", "label": "entailment", "score": 1.5},
             "invalid NLI score"),
            ({"status": "ready", "label": "ENTAILMENT", "score": 0.99},
             "unexpected NLI label"),
            ({"detail": "no output"}, "NLI inference probe failed"),
        ]
        for result, expected in cases:
            pipeline.verifier.nli_probe = lambda _result=result: _result
            health = security_health(pipeline)
            probe = health["readiness_probes"]["nli_verifier"]
            assert probe["status"] == "failed"
            assert expected in probe["detail"]
            assert health["status"] == DEGRADED
            assert health["research_ready"] is False
    finally:
        pipeline.close()


def test_nli_mode_that_is_not_loaded_is_failed_not_unverified(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        class _NotLoaded:
            mode = "nli"
            nli_ready = False

        pipeline.verifier = _NotLoaded()
        health = security_health(pipeline)
        probe = health["readiness_probes"]["nli_verifier"]
        assert probe["status"] == "failed"
        assert "not loaded" in probe["detail"]
        assert health["status"] == DEGRADED
        assert health["research_ready"] is False
    finally:
        pipeline.close()


def test_lexical_fallback_when_nli_is_required_is_degraded(config, monkeypatch):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        _ready_pipeline(pipeline, monkeypatch)
        pipeline.verifier = _FakeNLI(ready=False)
        pipeline.config.security.verifier = "nli"
        health = security_health(pipeline)
        assert health["readiness_probes"]["nli_verifier"]["status"] == "failed"
        assert health["status"] == DEGRADED
        assert health["research_ready"] is False
    finally:
        pipeline.close()


def test_unavailable_trust_ledger_is_unprotected(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        pipeline.trust_provider = NullTrustProvider()
        health = security_health(pipeline)
        assert health["readiness_probes"]["trust_ledger"]["status"] == "failed"
        assert health["status"] == UNPROTECTED
        assert health["research_ready"] is False
    finally:
        pipeline.close()


def test_absent_queue_is_unprotected(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        pipeline.security.queue = None
        health = security_health(pipeline)
        assert health["queue_enabled"] is False
        assert health["readiness_probes"]["queue"]["status"] == "failed"
        assert health["status"] == UNPROTECTED
    finally:
        pipeline.close()


def test_configured_model_names_alone_never_earn_protected(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    try:
        pipeline.config.embedding.model_name = "facebook/contriever"
        pipeline.config.generation.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        pipeline.config.security.verifier = "nli"
        health = security_health(pipeline)
        assert health["status"] == DEGRADED
        assert health["research_ready"] is False
        assert health["readiness_probes"]["nli_verifier"]["status"] == "failed"
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
