"""CachedLLM keeps cost/latency honest and never crosses prompts or models."""
from __future__ import annotations

import pytest

from trace_rag.config import GenerationConfig
from trace_rag.contracts import LLMResponse
from trace_rag.generation.llm import CachedLLM, StubLLM, build_llm, supports_chat_roles


class _Counting:
    name = "counting"
    model_name = "m1"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, max_tokens=256, stop=None):
        self.calls += 1
        return LLMResponse(text=f"echo:{prompt}", llm_calls=1, latency_ms=12.5, model=self.name)


def test_hit_returns_original_latency_and_counts_as_call(tmp_path):
    inner = _Counting()
    llm = CachedLLM(inner, str(tmp_path / "c.sqlite"))
    first = llm.generate("q", max_tokens=32)
    second = llm.generate("q", max_tokens=32)
    assert inner.calls == 1 and (llm.hits, llm.misses) == (1, 1)
    assert second.text == first.text
    assert second.llm_calls == 1 and second.latency_ms == 12.5
    llm.close()


def test_key_separates_prompt_max_tokens_stop_and_model(tmp_path):
    inner = _Counting()
    llm = CachedLLM(inner, str(tmp_path / "c.sqlite"))
    llm.generate("a", max_tokens=32)
    llm.generate("b", max_tokens=32)
    llm.generate("a", max_tokens=64)
    llm.generate("a", max_tokens=32, stop=["x"])
    inner.model_name = "m2"
    llm.generate("a", max_tokens=32)
    assert inner.calls == 5 and llm.hits == 0
    llm.close()


def test_chat_is_cached_only_when_backend_supports_it(tmp_path):
    assert not supports_chat_roles(CachedLLM(_Counting(), str(tmp_path / "a.sqlite")))
    llm = CachedLLM(StubLLM(), str(tmp_path / "b.sqlite"))
    assert supports_chat_roles(llm)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    a = llm.generate_chat(msgs, max_tokens=32)
    b = llm.generate_chat(msgs, max_tokens=32)
    assert (llm.hits, llm.misses) == (1, 1) and a.text == b.text
    llm.close()


def test_build_llm_wraps_only_greedy(tmp_path):
    cfg = GenerationConfig(cache_path=str(tmp_path / "c.sqlite"))
    assert isinstance(build_llm(cfg), CachedLLM)
    with pytest.raises(ValueError):
        build_llm(GenerationConfig(cache_path=str(tmp_path / "d.sqlite"), temperature=0.7))
    assert isinstance(build_llm(GenerationConfig()), StubLLM)
