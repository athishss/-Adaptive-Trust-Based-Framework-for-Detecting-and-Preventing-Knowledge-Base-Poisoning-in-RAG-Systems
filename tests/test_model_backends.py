"""Tests for the optional model backends.

These are the paths that need torch / transformers / a served model.  They are
exercised here without downloading anything: a tiny model is built locally and
the HTTP backends talk to a stub server that mimics the vLLM and Ollama APIs.

They skip when the optional dependencies are absent, and run on any machine
that has them (which is where the real experiments happen).
"""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

from trace_rag.contracts import LLMResponse  # noqa: E402
from trace_rag.generation.llm import LLMError, OllamaLLM, OpenAICompatLLM  # noqa: E402


@pytest.fixture(scope="module")
def tiny_encoder(tmp_path_factory) -> str:
    """A randomly initialised BERT + tokenizer saved locally (no downloads)."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    path = tmp_path_factory.mktemp("tiny_encoder")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(40)] + [
        "eiffel", "tower", "paris", "designed", "who", "gustave", "the", "in", "was", "by"]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_hf_embedder_produces_normalised_deterministic_vectors(tiny_encoder, pooling):
    from trace_rag.embeddings.hf_embedder import HFEmbedder

    embedder = HFEmbedder(model_name=tiny_encoder, device="cpu", pooling=pooling,
                          query_prefix="query: ", document_prefix="passage: ")
    texts = ["the eiffel tower in paris", "who designed the tower"]
    vectors = embedder.encode_documents(texts)
    assert vectors.shape == (2, embedder.dim) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.isfinite(vectors).all()
    assert np.allclose(vectors, embedder.encode_documents(texts), atol=1e-6)   # deterministic
    assert embedder.encode_queries(["who designed the tower"]).shape == (1, embedder.dim)


def test_hf_embedder_handles_padding_correctly(tiny_encoder):
    """A short text must embed identically alone and batched with a longer one."""
    from trace_rag.embeddings.hf_embedder import HFEmbedder

    embedder = HFEmbedder(model_name=tiny_encoder, device="cpu", pooling="mean")
    alone = embedder.encode_documents(["the eiffel tower in paris"])[0]
    batched = embedder.encode_documents(
        ["the eiffel tower in paris", "the eiffel tower in paris designed by gustave tok1 tok2"])[0]
    assert np.abs(alone - batched).max() < 1e-4, "attention mask is being mishandled"


def test_hf_embedder_batch_size_does_not_change_results(tiny_encoder):
    from trace_rag.embeddings.hf_embedder import HFEmbedder

    embedder = HFEmbedder(model_name=tiny_encoder, device="cpu")
    texts = ["the eiffel tower in paris"] * 5
    assert np.allclose(embedder.encode_documents(texts, batch_size=1),
                       embedder.encode_documents(texts, batch_size=5), atol=1e-5)


def test_pipeline_runs_with_a_real_transformer_embedder(tiny_encoder, tmp_path):
    """End to end with a genuine transformer in place of the hashing stand-in."""
    from trace_rag import Config, PersonAPipeline
    from trace_rag.embeddings.hf_embedder import HFEmbedder
    from trace_rag.index import NumpyFlatIndex
    from trace_rag.ingestion import Ingestor, ProvenanceStore

    config = Config.load(None, storage={"root": str(tmp_path / "run")})
    embedder = HFEmbedder(model_name=tiny_encoder, device="cpu")
    store = ProvenanceStore(config.path(config.storage.provenance_db))
    pipeline = PersonAPipeline(config=config, store=store, index=NumpyFlatIndex(embedder.dim),
                               embedder=embedder)
    ingestor = Ingestor(store, config.ingestion)
    ingestor.ingest_text("d1", "gustave designed the eiffel tower in paris", "wiki",
                         ingested_at=1.0, passage_mode=True)
    ingestor.ingest_text("d2", "who designed the eiffel tower the tower was designed by tok1",
                         "attacker", ingested_at=2.0, passage_mode=True)
    pipeline.index_chunks()
    result = pipeline.answer("who designed the eiffel tower", "q1")
    assert result.retrieval.documents and result.assessments
    assert all(0.0 <= a.suspicion <= 1.0 for a in result.assessments)
    pipeline.close()


# --------------------------------------------------------------------------
# HTTP backends against a stub server
# --------------------------------------------------------------------------

class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):        # silence the test output
        pass

    def do_POST(self):                   # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.last_payload = payload          # type: ignore[attr-defined]
        self.server.last_path = self.path           # type: ignore[attr-defined]
        if self.server.fail_with:                   # type: ignore[attr-defined]
            self.send_response(self.server.fail_with)  # type: ignore[attr-defined]
            self.end_headers()
            self.wfile.write(b'{"error": "boom"}')
            return
        if self.path.endswith("/chat/completions"):
            body = {"choices": [{"message": {"content": "Gustave Eiffel did. [d1#0000]"}}]}
        elif self.path.endswith("/api/generate"):
            body = {"response": "Gustave Eiffel did. [d1#0000]"}
        else:
            body = {}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    server.last_payload = None
    server.last_path = None
    server.fail_with = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_openai_compatible_backend_request_and_response(stub_server):
    port = stub_server.server_address[1]
    llm = OpenAICompatLLM("meta-llama/Llama-3.1-8B-Instruct", f"http://127.0.0.1:{port}/v1",
                          temperature=0.0, seed=123)
    response = llm.generate("Question: who designed it?", max_tokens=64, stop=["\n\n"])
    assert isinstance(response, LLMResponse)
    assert response.text == "Gustave Eiffel did. [d1#0000]" and response.llm_calls == 1
    payload = stub_server.last_payload
    assert stub_server.last_path.endswith("/v1/chat/completions")
    assert payload["temperature"] == 0.0 and payload["seed"] == 123     # greedy + reproducible
    assert payload["max_tokens"] == 64 and payload["stop"] == ["\n\n"]
    assert payload["messages"][0]["content"].startswith("Question:")


def test_ollama_backend_request_and_response(stub_server):
    port = stub_server.server_address[1]
    llm = OllamaLLM("llama3.1:8b", f"http://127.0.0.1:{port}", temperature=0.0, seed=7)
    response = llm.generate("Question: who designed it?", max_tokens=32)
    assert response.text == "Gustave Eiffel did. [d1#0000]"
    payload = stub_server.last_payload
    assert stub_server.last_path.endswith("/api/generate")
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0 and payload["options"]["seed"] == 7
    assert payload["options"]["num_predict"] == 32


def test_http_backends_raise_a_clear_error_on_server_failure(stub_server):
    stub_server.fail_with = 500
    port = stub_server.server_address[1]
    with pytest.raises(LLMError, match="HTTP 500"):
        OpenAICompatLLM("m", f"http://127.0.0.1:{port}/v1").generate("hi")


def test_http_backend_reports_unreachable_server():
    with pytest.raises(LLMError, match="cannot reach"):
        OpenAICompatLLM("m", "http://127.0.0.1:9/v1", timeout_s=2.0).generate("hi")


def test_grounded_generation_over_a_served_model(stub_server, tmp_path):
    """The served answer must flow through citation parsing and the answer log."""
    from trace_rag.config import GenerationConfig
    from trace_rag.generation import GroundedGenerator
    from conftest import make_doc

    port = stub_server.server_address[1]
    llm = OpenAICompatLLM("m", f"http://127.0.0.1:{port}/v1")
    docs = [make_doc("d1#0000", "Gustave Eiffel designed the tower.", trust=0.9)]
    outcome = GroundedGenerator(llm, GenerationConfig()).generate("who designed it?", "q1", docs)
    assert not outcome.record.abstained
    assert outcome.record.used_doc_ids == ("d1#0000",)
    assert outcome.record.evidence_mass == pytest.approx(0.9)


def test_hf_local_llm_generates_from_a_locally_built_model(tmp_path):
    """Exercises HFLocalLLM without downloading weights."""
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    from trace_rag.generation.llm import HFLocalLLM

    path = tmp_path / "tiny_causal"
    path.mkdir()
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{f"tok{i}": i + 4 for i in range(40)}}
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    from transformers import BertTokenizerFast

    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    config = GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(config).save_pretrained(path)

    llm = HFLocalLLM(model_name=str(path), device="cpu", dtype="float32", seed=1)
    response = llm.generate("tok1 tok2", max_tokens=5)
    assert isinstance(response.text, str) and response.llm_calls == 1
    assert response.latency_ms >= 0.0


def test_requesting_cuda_without_cuda_gives_an_actionable_error(tiny_encoder):
    """The raw torch error ('Torch not compiled with CUDA enabled') helps nobody."""
    from trace_rag.embeddings.hf_embedder import HFEmbedder, resolve_device

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA; the failure path cannot be exercised")
    with pytest.raises(RuntimeError) as error:
        HFEmbedder(model_name=tiny_encoder, device="cuda")
    message = str(error.value)
    assert "torch.cuda.is_available() is False" in message
    assert "--set embedding.device=cpu" in message      # tells the user what to type
    assert "download.pytorch.org" in message


def test_device_auto_falls_back_to_cpu_when_no_gpu(tiny_encoder):
    from trace_rag.embeddings.hf_embedder import HFEmbedder

    embedder = HFEmbedder(model_name=tiny_encoder, device="auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert embedder.device == expected
    assert embedder.encode_documents(["hello tok1"]).shape[0] == 1


def test_third_party_logging_is_quiet_by_default():
    import logging

    from trace_rag.utils.logging import get_logger

    get_logger("trace_rag.test")
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("huggingface_hub").level >= logging.WARNING
