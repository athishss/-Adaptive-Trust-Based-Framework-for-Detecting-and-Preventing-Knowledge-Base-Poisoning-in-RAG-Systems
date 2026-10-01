"""Regression tests for defects found during the verification pass.

Each test here failed against an earlier commit.  Keep them.
"""

from __future__ import annotations

import numpy as np
import pytest

from trace_rag.config import IngestionConfig
from trace_rag.contracts import FeatureSnapshot, SignalVector
from trace_rag.detection import LabelledRow, TrainingSet, leave_one_attack_out
from trace_rag.embeddings import HashingEmbedder
from trace_rag.generation import parse_citations
from trace_rag.index import NumpyFlatIndex
from trace_rag.ingestion import Ingestor, ProvenanceStore
from trace_rag.utils.hashing import build_family_assigner
from conftest import make_doc

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")
from trace_rag.index.faiss_index import FaissIndex  # noqa: E402


# --- 1. FAISS kept a stale vector when a passage's content changed -----------

@pytest.mark.parametrize("factory", [lambda d: NumpyFlatIndex(d), lambda d: FaissIndex(d, "flat")])
def test_reindexing_changed_content_updates_the_vector(factory):
    embedder = HashingEmbedder(dim=64)
    index = factory(64)
    old = embedder.encode_documents(["alpha alpha alpha"])
    new = embedder.encode_documents(["beta beta beta entirely different content"])
    index.add(["d#0000"], old)
    index.add(["d#0000"], new)
    assert np.allclose(index.get_vector("d#0000"), new[0], atol=1e-4)
    assert len(index) == 1
    top = index.search(new, 1)[0]
    assert top[0][0] == "d#0000" and top[0][1] > 0.99


def test_faiss_tombstones_are_not_returned_and_survive_reload(tmp_path):
    embedder = HashingEmbedder(dim=64)
    index = FaissIndex(64, "flat")
    index.add(["a", "b"], embedder.encode_documents(["first text about alpha", "text about beta"]))
    index.add(["a"], embedder.encode_documents(["rewritten text about gamma"]))
    assert index.dead_rows == 1 and len(index) == 2
    hits = index.search(embedder.encode_documents(["alpha"]), 5)[0]
    assert [h[0] for h in hits].count("a") == 1        # the tombstone is not a second hit

    index.save(tmp_path / "idx")
    reloaded = FaissIndex.load(tmp_path / "idx")
    assert len(reloaded) == 2 and reloaded.dead_rows == 1
    assert np.allclose(reloaded.get_vector("a"), index.get_vector("a"), atol=1e-5)


def test_faiss_ivfpq_and_hnsw_round_trip():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3000, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(3000)]
    for index in (FaissIndex(64, "ivfpq", nlist=32, pq_m=16), FaissIndex(64, "hnsw", hnsw_m=16)):
        index.add(ids, vectors)
        hits = index.search(vectors[:1], 5)[0]
        assert hits and hits[0][0] == "c0"


def test_ivfpq_refuses_impossible_training_size():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(10, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="nlist"):
        FaissIndex(64, "ivfpq", nlist=256, pq_m=16).add([f"c{i}" for i in range(10)], vectors)


# --- 2. leave-one-attack-out leaked clean rows and questions ------------------

def _loao_dataset(n_questions: int = 30) -> TrainingSet:
    dataset = TrainingSet()
    families = ["fam_a", "fam_b", "fam_c"]
    for q in range(n_questions):
        family = families[q % 3]
        for i in range(3):
            dataset.add(LabelledRow(f"q{q}", f"q{q}_c{i}", 0, FeatureSnapshot(SignalVector(), {}), "none"))
        dataset.add(LabelledRow(f"q{q}", f"q{q}_p", 1, FeatureSnapshot(SignalVector(), {}), family))
    return dataset


def test_leave_one_attack_out_shares_no_rows_or_questions():
    dataset = _loao_dataset()
    folds = list(leave_one_attack_out(dataset.families, dataset.query_ids))
    assert len(folds) == 3
    for held_out, train_idx, test_idx in folds:
        assert not set(train_idx) & set(test_idx)
        train_questions = {dataset.query_ids[i] for i in train_idx}
        test_questions = {dataset.query_ids[i] for i in test_idx}
        assert train_questions.isdisjoint(test_questions)
        assert {dataset.families[i] for i in test_idx} <= {held_out, "none"}
        assert held_out not in {dataset.families[i] for i in train_idx}
        assert any(dataset.y[i] == 0 for i in test_idx)   # test side keeps an FPR sample


def test_leave_one_attack_out_leaky_mode_is_opt_in():
    dataset = _loao_dataset()
    _, train_idx, test_idx = next(leave_one_attack_out(dataset.families, dataset.query_ids,
                                                        include_clean="always"))
    assert set(train_idx) & set(test_idx)               # documented, deliberate


def test_leave_one_attack_out_validates_arguments():
    with pytest.raises(ValueError):
        list(leave_one_attack_out(["none", "fam_a"], ["q1", "q2"]))
    with pytest.raises(ValueError):
        list(leave_one_attack_out(["fam_a", "fam_b"], ["q1"]))
    with pytest.raises(ValueError):
        list(leave_one_attack_out(["fam_a", "fam_b"], None, include_clean="nonsense"))


# --- 3. empty brackets survived citation cleanup ------------------------------

def test_empty_and_invalid_brackets_are_cleaned():
    citations, invalid, cleaned = parse_citations(
        "Claim one [d1]. Odd [../etc/passwd]. Empty []. Done.", [make_doc("d1")])
    assert [c.doc_id for c in citations] == ["d1"]
    assert invalid == ["../etc/passwd"]
    assert "[" not in cleaned and "]" not in cleaned


# --- 4. source counters were recomputed with COUNT(*) (quadratic) -------------

def test_source_counters_stay_correct_through_reingestion(store):
    ingestor = Ingestor(store, IngestionConfig())
    ingestor.ingest_text("d1", "word " * 300, "src_a", ingested_at=1.0)
    ingestor.ingest_text("d2", "other " * 300, "src_a", ingested_at=2.0)
    stats = store.source_stats("src_a")
    assert stats.n_docs == 2 and stats.n_chunks == store.counts()["chunks"]

    ingestor.ingest_text("d1", "short replacement text", "src_a", ingested_at=3.0, passage_mode=True)
    stats = store.source_stats("src_a")
    assert stats.n_docs == 2 and stats.n_chunks == store.counts()["chunks"]

    ingestor.ingest_text("d1", "moved to another contributor entirely", "src_b", ingested_at=4.0,
                         passage_mode=True)
    a, b = store.source_stats("src_a"), store.source_stats("src_b")
    assert a.n_docs == 1 and b.n_docs == 1
    assert a.n_chunks + b.n_chunks == store.counts()["chunks"]


def test_ingestion_of_a_thousand_passages_is_not_quadratic(store):
    """Guards the COUNT(*)-per-document regression; generous bound for slow CI."""
    import time

    ingestor = Ingestor(store, IngestionConfig(family_backend="exact"))
    start = time.perf_counter()
    with store.batch():
        for i in range(1000):
            ingestor.ingest_text(f"d{i}", f"passage {i} about subject {i % 50}", f"src{i % 20}",
                                 ingested_at=1000.0 + i, passage_mode=True)
    elapsed = time.perf_counter() - start
    assert store.counts()["chunks"] == 1000
    assert elapsed < 20.0, f"1000 passages took {elapsed:.1f}s - counters may be quadratic again"


def test_batch_rolls_back_on_error(tmp_path):
    store = ProvenanceStore(tmp_path / "p.sqlite3")
    ingestor = Ingestor(store, IngestionConfig())
    with pytest.raises(RuntimeError):
        with store.batch():
            ingestor.ingest_text("d1", "some passage text", "src", ingested_at=1.0, passage_mode=True)
            raise RuntimeError("boom")
    assert store.counts()["chunks"] == 0
    store.close()


# --- 5. near-duplicate detection had unbounded cost and memory ----------------

@pytest.mark.parametrize("backend,expect_near_dup", [("minhash", True), ("exact", False),
                                                     ("none", False)])
def test_family_backends(backend, expect_near_dup):
    assigner = build_family_assigner(backend)
    base = "The Eiffel Tower was completed in Paris in 1889 for the World Fair exhibition"
    near = "The Eiffel Tower was completed in Paris in 1889 for the World Fair exposition"
    fam_a = assigner.assign("c1", base)
    fam_b = assigner.assign("c2", near)
    fam_exact = assigner.assign("c3", base)
    assert (fam_a == fam_b) is expect_near_dup
    assert (fam_a == fam_exact) is (backend != "none")


def test_minhash_candidate_cap_keeps_assignment_deterministic():
    texts = [f"the historical record shows several notable developments in area {i % 3}"
             for i in range(200)]
    first = build_family_assigner("minhash", max_candidates=8)
    second = build_family_assigner("minhash", max_candidates=8)
    assert [first.assign(f"c{i}", t) for i, t in enumerate(texts)] == \
           [second.assign(f"c{i}", t) for i, t in enumerate(texts)]


def test_unknown_family_backend_rejected():
    with pytest.raises(ValueError):
        build_family_assigner("magic")


# --- 7. work package A7: profiling and ablation must run end to end ----------

def test_profile_and_ablate_script(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import profile_and_ablate as a7

    out = tmp_path / "a7.json"
    monkeypatch.setattr(sys, "argv", ["a7", "--root", str(tmp_path / "run"),
                                      "--repeats", "1", "--out", str(out)])
    assert a7.main() == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "total" in report["latency"] and report["latency"]["total"]["mean_ms"] > 0
    assert report["latency"]["llm_calls_per_query"]["mean"] >= 1
    assert "all_signals" in report["ablation"]
    assert all(f"without_{s}" in report["ablation"] for s in
               ("s1_query_echo", "s6_neighbourhood_density"))


# --- 8. Hugging Face serves Parquet, not JSONL ------------------------------

def test_beir_parquet_is_readable(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from trace_rag.ingestion import iter_beir_corpus

    table = pa.table({"_id": pa.array([f"doc{i}" for i in range(50)]),
                      "title": pa.array([f"Title {i}" for i in range(50)]),
                      "text": pa.array([f"Passage {i} of encyclopaedic prose." for i in range(50)])})
    path = tmp_path / "corpus-00000-of-00001.parquet"
    pq.write_table(table, path)
    rows = list(iter_beir_corpus(path))
    assert len(rows) == 50
    assert rows[0] == {"doc_id": "doc0", "title": "Title 0",
                       "text": "Passage 0 of encyclopaedic prose."}


def test_beir_parquet_without_id_column_fails_loudly(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from trace_rag.ingestion import ParseError, iter_beir_corpus

    pq.write_table(pa.table({"text": pa.array(["a", "b"])}), tmp_path / "bad.parquet")
    with pytest.raises(ParseError, match="_id"):
        list(iter_beir_corpus(tmp_path / "bad.parquet"))


def test_unsupported_corpus_suffix_rejected(tmp_path):
    from trace_rag.ingestion import ParseError, iter_beir_corpus

    (tmp_path / "corpus.csv").write_text("a,b\n1,2", encoding="utf-8")
    with pytest.raises(ParseError):
        list(iter_beir_corpus(tmp_path / "corpus.csv"))


# --- 9. chunk ids collided for different documents --------------------------

def test_chunk_ids_do_not_collide():
    from trace_rag.ingestion import make_chunk_id

    long_a, long_b = "file::" + "x" * 50 + "::AAAA", "file::" + "x" * 50 + "::BBBB"
    assert make_chunk_id(long_a, 0) != make_chunk_id(long_b, 0)
    assert len({make_chunk_id(x, 0) for x in ("doc/one", "doc one", "doc_one")}) == 3
    # benchmark-style ids stay readable
    assert make_chunk_id("nq_1", 0) == "nq_1#0000"
    assert make_chunk_id("poison_eiffel_0", 3) == "poison_eiffel_0#0003"


def test_chunk_ids_are_unique_under_fuzzing():
    import random
    import string

    from trace_rag.ingestion import make_chunk_id

    rng = random.Random(7)
    alphabet = string.ascii_letters + string.digits + "/\\ .:-_#~%?&=+" + "éü漢字"
    doc_ids = {"".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80))) for _ in range(5000)}
    doc_ids |= {"a" * 60 + "1", "a" * 60 + "2", "x/y", "x y", "x_y", "~", "a~b"}
    doc_ids.discard("")
    mapped = {make_chunk_id(d, 0) for d in doc_ids}
    assert len(mapped) == len(doc_ids)


def test_colliding_document_ids_keep_separate_rows(store):
    """The collision used to overwrite one document's chunks with another's."""
    ingestor = Ingestor(store, IngestionConfig())
    ingestor.ingest_text("report/2026", "First document about alpha topics", "src",
                         ingested_at=1.0, passage_mode=True)
    ingestor.ingest_text("report 2026", "Second and unrelated document about beta", "src",
                         ingested_at=2.0, passage_mode=True)
    assert store.counts()["documents"] == 2
    assert store.counts()["chunks"] == 2
    texts = {c.text for c in store.iter_chunks()}
    assert len(texts) == 2


# --- 10. dataset download and subset preparation ----------------------------

def _load_download_script():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import download_data

    return download_data


def test_qrels_parsing_ignores_non_relevant_rows(tmp_path):
    download_data = _load_download_script()
    (tmp_path / "qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq0\tdocA\t1\nq0\tdocB\t0\nq1\tdocC\t2\n", encoding="utf-8")
    gold = download_data.read_qrels(tmp_path / "qrels.tsv")
    assert gold == {"q0": {"docA"}, "q1": {"docC"}}


def test_subset_always_keeps_gold_passages(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    download_data = _load_download_script()
    ids = [f"doc{i}" for i in range(2000)]
    pq.write_table(pa.table({"_id": pa.array(ids),
                             "title": pa.array([f"t{i}" for i in range(2000)]),
                             "text": pa.array([f"passage {i}" for i in range(2000)])}),
                   tmp_path / "corpus.parquet")
    gold_ids = {f"doc{i}" for i in range(0, 2000, 97)}      # scattered through the file
    kept, gold_kept = download_data.build_subset(tmp_path / "corpus.parquet", gold_ids, 300,
                                                 tmp_path / "subset.parquet")
    assert gold_kept == len(gold_ids)                        # never drop a gold passage
    assert 0.5 * 300 <= kept <= 1.6 * 300                    # roughly the requested size

    from trace_rag.ingestion import iter_beir_corpus

    rows = list(iter_beir_corpus(tmp_path / "subset.parquet"))
    assert len(rows) == kept
    assert gold_ids <= {r["doc_id"] for r in rows}


def test_subset_is_deterministic(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    download_data = _load_download_script()
    pq.write_table(pa.table({"_id": pa.array([f"d{i}" for i in range(500)]),
                             "title": pa.array([""] * 500),
                             "text": pa.array([f"p{i}" for i in range(500)])}),
                   tmp_path / "c.parquet")
    first = download_data.build_subset(tmp_path / "c.parquet", {"d1"}, 100, tmp_path / "a.parquet")
    second = download_data.build_subset(tmp_path / "c.parquet", {"d1"}, 100, tmp_path / "b.parquet")
    assert first == second


def test_gpu_config_is_valid():
    from pathlib import Path

    from trace_rag.config import Config

    config = Config.load(Path(__file__).resolve().parent.parent / "config" / "nq_gpu.yaml")
    assert config.embedding.backend == "huggingface"
    assert config.embedding.device in {"auto", "cuda", "cpu"}
    assert config.index.backend == "faiss"
    assert config.ingestion.family_backend == "exact"     # memory-safe at corpus scale


def test_indexing_streams_instead_of_loading_everything(config, monkeypatch):
    """index_chunks must not materialise the whole store in memory."""
    from trace_rag.ingestion import Ingestor
    from trace_rag.pipeline import PersonAPipeline

    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    ingestor = Ingestor(pipeline.store, config.ingestion)
    with pipeline.store.batch():
        for i in range(500):
            ingestor.ingest_text(f"d{i}", f"passage {i} about subject {i % 7}", f"s{i % 5}",
                                 ingested_at=1_600_000_000.0 + i, passage_mode=True)

    calls = {"iter": 0, "list": 0}
    original_iter = pipeline.store.iter_chunks

    def counting_iter(*args, **kwargs):
        calls["iter"] += 1
        return original_iter(*args, **kwargs)

    monkeypatch.setattr(pipeline.store, "iter_chunks", counting_iter)
    assert pipeline.index_chunks(batch_size=64) == 500
    assert calls["iter"] >= 1
    assert len(pipeline.index) == 500
    pipeline.close()


def test_indexing_reports_progress_regularly(config, caplog):
    """A long index run must not look frozen; progress is logged per block."""
    import logging

    from trace_rag.ingestion import Ingestor
    from trace_rag.pipeline import PersonAPipeline

    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    ingestor = Ingestor(pipeline.store, config.ingestion)
    with pipeline.store.batch():
        for i in range(600):
            ingestor.ingest_text(f"d{i}", f"passage {i} about subject {i % 9}", f"s{i % 4}",
                                 ingested_at=1_600_000_000.0 + i, passage_mode=True)
    with caplog.at_level(logging.INFO, logger="trace_rag.pipeline"):
        pipeline.index_chunks(batch_size=50, progress_every=100)
    messages = [record.message for record in caplog.records]
    assert any("indexing" in m and "passages with" in m for m in messages)   # start banner
    progress = [m for m in messages if m.startswith("indexed ") and "/s," in m]
    assert len(progress) >= 5, f"expected regular progress lines, got {progress}"
    pipeline.close()


# --- 11. OFFSET pagination made streaming slower the further it went --------

def test_iter_chunks_returns_every_row_once_in_order(store):
    ingestor = Ingestor(store, IngestionConfig(family_backend="exact"))
    with store.batch():
        for i in range(2500):
            ingestor.ingest_text(f"doc{i:05d}", f"passage number {i}", f"s{i % 7}",
                                 ingested_at=1_600_000_000.0 + i, passage_mode=True)
    seen = [record.chunk_id for record in store.iter_chunks(batch_size=97)]
    assert len(seen) == 2500
    assert len(set(seen)) == 2500                 # no duplicates from paging
    assert seen == sorted(seen)                   # stable order
    assert set(seen) == {f"doc{i:05d}#0000" for i in range(2500)}


def test_iter_chunks_does_not_slow_down_with_depth(store):
    """Keyset paging: the last slice must not cost much more than the first."""
    import time

    ingestor = Ingestor(store, IngestionConfig(family_backend="exact"))
    with store.batch():
        for i in range(40000):
            ingestor.ingest_text(f"doc{i:06d}", f"passage {i}", f"s{i % 50}",
                                 ingested_at=1_600_000_000.0 + i, passage_mode=True)
    marks = {}
    start = time.perf_counter()
    for n, _ in enumerate(store.iter_chunks(batch_size=1000), start=1):
        if n in (10000, 40000):
            marks[n] = time.perf_counter() - start
    first_quarter = marks[10000]
    last_quarter = marks[40000] - marks[10000] * 3
    assert last_quarter < first_quarter * 3 + 0.05, (
        f"streaming degrades with depth: first 10k {first_quarter:.2f}s, "
        f"last 10k {last_quarter:.2f}s")


# --- 12. indexing lost everything if the machine died mid-run ---------------

def test_indexing_resumes_and_skips_what_is_done(config):
    """A crash, a power cut or Ctrl+C must not cost the whole run."""
    from trace_rag.ingestion import Ingestor
    from trace_rag.pipeline import PersonAPipeline

    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    ingestor = Ingestor(pipeline.store, config.ingestion)
    with pipeline.store.batch():
        for i in range(1000):
            ingestor.ingest_text(f"d{i:04d}", f"passage {i} about subject {i % 9}", f"s{i % 4}",
                                 ingested_at=1_600_000_000.0 + i, passage_mode=True)

    first = pipeline.index_chunks(batch_size=50, limit=300, save_every=100)
    assert 300 <= first <= 350            # stops at the first batch boundary past the limit
    indexed_after_first = len(pipeline.index)
    pipeline.save_index()
    pipeline.close()

    resumed = PersonAPipeline.from_config(config, load_existing_index=True)
    assert len(resumed.index) == indexed_after_first      # checkpoint survived the restart
    second = resumed.index_chunks(batch_size=50)
    assert second == 1000 - indexed_after_first           # only the remainder was embedded
    assert len(resumed.index) == 1000
    third = resumed.index_chunks(batch_size=50)
    assert third == 0                                     # nothing left to do
    resumed.close()


def test_index_limit_lets_you_work_in_sittings(config):
    from trace_rag.ingestion import Ingestor
    from trace_rag.pipeline import PersonAPipeline

    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    ingestor = Ingestor(pipeline.store, config.ingestion)
    with pipeline.store.batch():
        for i in range(400):
            ingestor.ingest_text(f"d{i:04d}", f"passage {i}", "s", ingested_at=1_600_000_000.0 + i,
                                 passage_mode=True)
    assert pipeline.index_chunks(batch_size=25, limit=100) == 100
    assert pipeline.index_chunks(batch_size=25, limit=100) == 100
    assert len(pipeline.index) == 200
    pipeline.close()
