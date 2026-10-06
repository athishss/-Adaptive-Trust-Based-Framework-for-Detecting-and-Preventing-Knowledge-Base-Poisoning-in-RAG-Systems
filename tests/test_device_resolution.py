"""Accelerator selection tests that run without torch, Transformers, or XLA."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from trace_rag.embeddings.hf_embedder import resolve_device


class _CudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _Torch:
    cuda = _CudaUnavailable()


def _install_xla(monkeypatch: pytest.MonkeyPatch, value: str = "xla:0") -> None:
    root = ModuleType("torch_xla")
    root.__path__ = []  # type: ignore[attr-defined]
    core = ModuleType("torch_xla.core")
    core.__path__ = []  # type: ignore[attr-defined]
    model = ModuleType("torch_xla.core.xla_model")
    model.xla_device = lambda: value  # type: ignore[attr-defined]
    root.core = core  # type: ignore[attr-defined]
    core.xla_model = model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_xla", root)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", model)


def test_explicit_tpu_device_resolves_through_pytorch_xla(monkeypatch):
    _install_xla(monkeypatch)
    assert resolve_device("tpu", _Torch()) == "xla:0"
    assert resolve_device("xla", _Torch()) == "xla:0"


def test_auto_prefers_visible_tpu_when_cuda_is_absent(monkeypatch):
    _install_xla(monkeypatch)
    assert resolve_device("auto", _Torch()) == "xla:0"


def test_tpu_request_without_an_xla_device_fails_actionably(monkeypatch):
    _install_xla(monkeypatch, value="cpu")
    with pytest.raises(RuntimeError, match="no TPU/XLA device"):
        resolve_device("tpu", _Torch())


def test_cpu_selection_does_not_require_optional_accelerator_packages():
    assert resolve_device("cpu", _Torch()) == "cpu"


def test_nli_pipeline_receives_premise_and_hypothesis_as_a_pair(monkeypatch):
    calls = []

    class FakePipeline:
        def __call__(self, value):
            calls.append(value)
            return [{"label": "entailment", "score": 0.9}]

    def pipeline_factory(task, **kwargs):  # noqa: ARG001
        return FakePipeline()

    torch = ModuleType("torch")
    torch.cuda = _CudaUnavailable()  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.pipeline = pipeline_factory  # type: ignore[attr-defined]
    monkeypatch.delenv("TRACE_RAG_NLI_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    from trace_rag.trust.verifier import NLIScorer

    scorer = NLIScorer()
    assert scorer.available
    assert scorer.device == "cpu"
    assert scorer.predict("A premise.", "A hypothesis.") == ("entailment", 0.9)
    assert calls == [
        {"text": "Test premise.", "text_pair": "Test hypothesis."},
        {"text": "A premise.", "text_pair": "A hypothesis."},
    ]


def test_generation_factory_passes_the_configured_device(monkeypatch):
    from trace_rag.config import GenerationConfig
    from trace_rag.generation import llm as llm_module

    captured = {}

    def fake_local_llm(model_name, device, seed):
        captured.update(model_name=model_name, device=device, seed=seed)
        return object()

    monkeypatch.setattr(llm_module, "HFLocalLLM", fake_local_llm)
    config = GenerationConfig(backend="huggingface", model_name="small-model",
                              device="tpu", seed=7)
    llm_module.build_llm(config)
    assert captured == {"model_name": "small-model", "device": "tpu", "seed": 7}


def test_embedder_factory_passes_the_configured_max_length(monkeypatch):
    from trace_rag.config import EmbeddingConfig
    from trace_rag.embeddings import build_embedder
    from trace_rag.embeddings import hf_embedder

    captured = {}

    def fake_embedder(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(hf_embedder, "HFEmbedder", fake_embedder)
    build_embedder(EmbeddingConfig(backend="huggingface", model_name="small-model",
                                  max_length=128, device="tpu"))
    assert captured["max_length"] == 128
    assert captured["device"] == "tpu"
