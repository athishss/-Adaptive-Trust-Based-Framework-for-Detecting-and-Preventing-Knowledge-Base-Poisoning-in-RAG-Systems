from __future__ import annotations

import numpy as np
import pytest

from trace_rag.config import RetrievalConfig
from trace_rag.contracts import TrustSnapshot, TrustStatus
from trace_rag.embeddings import HashingEmbedder
from trace_rag.index import NumpyFlatIndex
from trace_rag.retrieval import TrustWeightedRetriever
from conftest import random_unit_vectors

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")
from trace_rag.index.faiss_index import FaissIndex  # noqa: E402


@pytest.fixture
def vectors():
    return random_unit_vectors(300, 64, seed=3)


def test_numpy_and_faiss_agree_on_top_k(vectors):
    ids = [f"c{i}" for i in range(len(vectors))]
    numpy_index, faiss_index = NumpyFlatIndex(64), FaissIndex(64, "flat")
    numpy_index.add(ids, vectors)
    faiss_index.add(ids, vectors)
    for query in vectors[:5]:
        a = [doc_id for doc_id, _ in numpy_index.search(query[None, :], 10)[0]]
        b = [doc_id for doc_id, _ in faiss_index.search(query[None, :], 10)[0]]
        assert a == b


def test_index_excludes_blocked_ids(vectors):
    ids = [f"c{i}" for i in range(len(vectors))]
    for index in (NumpyFlatIndex(64), FaissIndex(64, "flat")):
        index.add(ids, vectors)
        hits = index.search(vectors[0][None, :], 5, exclude={"c0", "c1"})[0]
        assert {"c0", "c1"}.isdisjoint({doc_id for doc_id, _ in hits})
        assert len(hits) == 5


def test_index_roundtrip(tmp_path, vectors):
    ids = [f"c{i}" for i in range(len(vectors))]
    numpy_index = NumpyFlatIndex(64)
    numpy_index.add(ids, vectors)
    numpy_index.save(tmp_path / "idx")
    reloaded = NumpyFlatIndex.load(tmp_path / "idx")
    assert len(reloaded) == len(numpy_index)
    assert reloaded.search(vectors[0][None, :], 3)[0] == numpy_index.search(vectors[0][None, :], 3)[0]

    faiss_index = FaissIndex(64, "flat")
    faiss_index.add(ids, vectors)
    faiss_index.save(tmp_path / "fidx")
    reloaded_faiss = FaissIndex.load(tmp_path / "fidx")
    assert len(reloaded_faiss) == len(faiss_index)


def test_adding_same_id_twice_does_not_duplicate(vectors):
    index = NumpyFlatIndex(64)
    index.add(["a", "b"], vectors[:2])
    index.add(["a", "b"], vectors[:2])
    assert len(index) == 2


def test_index_rejects_wrong_dimension():
    index = NumpyFlatIndex(8)
    with pytest.raises(ValueError):
        index.add(["a"], np.zeros((1, 4), dtype=np.float32))


def test_empty_index_returns_empty_results():
    assert NumpyFlatIndex(8).search(np.zeros((1, 8), dtype=np.float32), 5) == [[]]


def _retriever(pipeline_store, index, embedder, trust_provider=None, **cfg):
    return TrustWeightedRetriever(pipeline_store, index, embedder,
                                  RetrievalConfig(**cfg), trust_provider)


def test_retrieval_returns_provenance(populated_pipeline):
    outcome = populated_pipeline.retrieve("Who designed the Eiffel Tower?", "q1")
    assert outcome.documents and len(outcome.documents) <= populated_pipeline.config.retrieval.top_k
    assert len(outcome.pool) >= len(outcome.documents)
    for doc in outcome.documents:
        assert doc.source_id and doc.family_id and doc.chunk_id
        assert 0.0 <= doc.trust.t_eff <= 1.0


def test_trust_weighting_reorders_results(populated_pipeline):
    class LowTrustForPoison:
        def get_trust(self, doc_ids):
            out = {}
            for doc_id in doc_ids:
                trust = 0.05 if doc_id.startswith("poison") else 0.9
                out[doc_id] = TrustSnapshot(doc_id, "s", "f", trust, trust, trust, trust)
            return out

        def blocked_doc_ids(self):
            return set()

    query = "Who designed the Eiffel Tower?"
    neutral = populated_pipeline.retrieve(query, "q1")
    populated_pipeline.retriever.trust = LowTrustForPoison()
    weighted = populated_pipeline.retrieve(query, "q2")

    def poison_rank(outcome):
        for i, doc in enumerate(outcome.documents):
            if doc.doc_id.startswith("poison"):
                return i
        return len(outcome.documents)

    assert poison_rank(weighted) >= poison_rank(neutral)


def test_quarantined_documents_are_never_retrieved(populated_pipeline):
    blocked = {"poison_1#0000", "poison_2#0000", "poison_3#0000"}

    class BlockingProvider:
        def get_trust(self, doc_ids):
            return {d: TrustSnapshot(d, "s", "f", 0.5, 0.5, 0.5, 0.5) for d in doc_ids}

        def blocked_doc_ids(self):
            return blocked

    populated_pipeline.retriever.trust = BlockingProvider()
    outcome = populated_pipeline.retrieve("Who designed the Eiffel Tower?", "q3")
    assert blocked.isdisjoint({d.doc_id for d in outcome.pool})


def test_trust_floor_keeps_new_sources_reachable(populated_pipeline):
    class ZeroTrust:
        def get_trust(self, doc_ids):
            return {d: TrustSnapshot(d, "s", "f", 0.0, 0.0, 0.0, 0.0) for d in doc_ids}

        def blocked_doc_ids(self):
            return set()

    populated_pipeline.retriever.trust = ZeroTrust()
    outcome = populated_pipeline.retrieve("Who designed the Eiffel Tower?", "q4")
    assert outcome.documents           # floor prevents total censorship of zero-trust content


def test_empty_query_rejected(populated_pipeline):
    with pytest.raises(ValueError):
        populated_pipeline.retrieve("   ", "q5")


def test_retrieval_is_deterministic(populated_pipeline):
    a = populated_pipeline.retrieve("Who designed the Eiffel Tower?", "q6").doc_ids()
    b = populated_pipeline.retrieve("Who designed the Eiffel Tower?", "q7").doc_ids()
    assert a == b
