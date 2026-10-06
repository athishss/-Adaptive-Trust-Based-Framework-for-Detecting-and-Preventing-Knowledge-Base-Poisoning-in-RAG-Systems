from __future__ import annotations

import json

import pytest

from trace_rag.cli import main
from trace_rag.config import Config


def test_config_defaults_and_roundtrip(tmp_path):
    config = Config()
    path = tmp_path / "config.yaml"
    config.dump(path)
    reloaded = Config.load(path)
    assert reloaded.model_dump() == config.model_dump()


def test_config_validation_rejects_bad_values():
    with pytest.raises(Exception):
        Config.load(None, ingestion={"chunk_words": 50, "chunk_overlap_words": 60})
    with pytest.raises(Exception):
        Config.load(None, scorer={"theta_low": 0.9, "theta_high": 0.2})
    with pytest.raises(Exception):
        Config.load(None, retrieval={"top_k": 0})
    with pytest.raises(Exception, match="w_r"):
        Config.load(None, trust={"w_s": 1.0, "w_r": 1.0})
    with pytest.raises(Exception, match="reject_refutations"):
        Config.load(None, trust={"quarantine_refutations": 3,
                                 "reject_refutations": 3})
    with pytest.raises(Exception):
        Config.load(None, generation={"citation_nli_threshold": 1.5})


def test_shipped_configs_parse_with_trust_settings():
    for path in ("config/default.yaml", "config/full_nq.yaml", "config/nq_gpu.yaml",
                 "config/nq_tpu.yaml"):
        config = Config.load(path)
        assert config.storage.trust_db
        if path != "config/default.yaml":
            assert config.trust.enabled
            assert config.generation.require_citations
    tpu = Config.load("config/nq_tpu.yaml")
    assert tpu.embedding.device == "tpu"
    assert tpu.index.backend == "faiss" and tpu.index.faiss_kind == "ivfpq"
    assert tpu.trust.nli_mode == "nli"
    assert tpu.generation.backend == "ollama"


def test_config_overrides_apply():
    config = Config.load(None, retrieval={"top_k": 9, "trust_lambda": 2.0})
    assert config.retrieval.top_k == 9 and config.retrieval.trust_lambda == 2.0


def _write_corpus(path):
    rows = [
        {"_id": "c1", "title": "Eiffel", "text": "Gustave Eiffel designed the Eiffel Tower in 1889."},
        {"_id": "c2", "title": "Plants", "text": "Photosynthesis happens inside chloroplasts."},
        {"_id": "c3", "title": "Everest", "text": "Mount Everest is 8849 metres high."},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_cli_end_to_end(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    root = str(tmp_path / "run")

    assert main(["--root", root, "ingest-beir", "--corpus", str(corpus), "--n-sources", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 3

    assert main(["--root", root, "index"]) == 0
    assert json.loads(capsys.readouterr().out)["indexed"] == 3

    assert main(["--root", root, "query", "Who designed the Eiffel Tower?", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieved"] and payload["answer"]["query"]

    assert main(["--root", root, "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["index_size"] == 3

    cited = payload["answer"]["used_doc_ids"] or ["c1#0000"]
    assert main(["--root", root, "remediate", "--doc-ids", *cited]) == 0
    assert "affected_answer_ids" in json.loads(capsys.readouterr().out)


def test_cli_ingest_files(tmp_path, capsys):
    docs = tmp_path / "docs" / "teamA"
    docs.mkdir(parents=True)
    (docs / "note.txt").write_text("The Eiffel Tower was completed in 1889.", encoding="utf-8")
    (docs / "page.md").write_text("# Tower\n\nDesigned by Gustave Eiffel.", encoding="utf-8")
    assert main(["--root", str(tmp_path / "run"), "ingest", "--path", str(tmp_path / "docs")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["documents"] == 2 and report["errors"] == []


def test_cli_train_scorer(tmp_path, capsys):
    rows_path = tmp_path / "rows.jsonl"
    lines = []
    for q in range(40):
        for i in range(4):
            lines.append(json.dumps({
                "query_id": f"q{q}", "doc_id": f"q{q}_clean{i}", "label": 0,
                "features": {"s1_query_echo": 0.1 + 0.01 * i, "s2_similarity_outlier": 0.2,
                             "s3_cluster_tightness": 0.1, "s4_ingestion_burst": 0.0,
                             "s5_source_immaturity": 0.1, "s6_neighbourhood_density": 0.1,
                             "x_similarity": 0.4, "x_rank": float(i), "x_source_log_age_days": 6.0,
                             "x_source_n_docs": 100.0, "x_source_trust": 0.8, "x_doc_trust": 0.8,
                             "x_family_size": 1.0}}))
        lines.append(json.dumps({
            "query_id": f"q{q}", "doc_id": f"q{q}_poison", "label": 1,
            "attack_family": "poisonedrag_bb",
            "features": {"s1_query_echo": 0.95, "s2_similarity_outlier": 0.8,
                         "s3_cluster_tightness": 0.7, "s4_ingestion_burst": 0.6,
                         "s5_source_immaturity": 0.9, "s6_neighbourhood_density": 0.6,
                         "x_similarity": 0.9, "x_rank": 0.0, "x_source_log_age_days": 0.1,
                         "x_source_n_docs": 3.0, "x_source_trust": 0.5, "x_doc_trust": 0.5,
                         "x_family_size": 3.0}}))
    rows_path.write_text("\n".join(lines), encoding="utf-8")

    out = tmp_path / "scorer.joblib"
    assert main(["train-scorer", "--rows", str(rows_path), "--out", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert out.exists()
    assert report["dataset"]["poison"] == 40
    assert report["held_out_test"]["roc_auc"] == 1.0          # trivially separable fixture


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "ingest-beir", "--corpus", str(tmp_path / "missing.jsonl")]) == 1


def test_set_overrides_apply(tmp_path, capsys):
    """--set lets you change any config value without editing a file."""
    from trace_rag.cli import apply_overrides
    from trace_rag.config import Config

    config = apply_overrides(Config(), ["generation.backend=stub", "retrieval.top_k=9",
                                        "embedding.device=cpu", "retrieval.trust_lambda=2.5"])
    assert config.generation.backend == "stub"
    assert config.retrieval.top_k == 9 and isinstance(config.retrieval.top_k, int)
    assert config.embedding.device == "cpu"
    assert config.retrieval.trust_lambda == 2.5


def test_set_override_rejects_unknown_keys():
    import pytest

    from trace_rag.cli import apply_overrides
    from trace_rag.config import Config

    with pytest.raises(ValueError, match="unknown config key"):
        apply_overrides(Config(), ["retrieval.nonsense=1"])
    with pytest.raises(ValueError, match="unknown config section"):
        apply_overrides(Config(), ["nope.key=1"])
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides(Config(), ["retrieval.top_k"])


def test_set_override_rejects_invalid_value():
    import pytest

    from trace_rag.cli import apply_overrides
    from trace_rag.config import Config

    with pytest.raises(Exception):
        apply_overrides(Config(), ["retrieval.top_k=0"])       # fails validation


def test_cli_query_with_overrides(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    root = str(tmp_path / "run")
    assert main(["--root", root, "ingest-beir", "--corpus", str(corpus), "--n-sources", "2"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "index"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "--set", "generation.backend=stub", "--set", "retrieval.top_k=2",
                 "query", "Who designed the Eiffel Tower?", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["retrieved"]) <= 2


def test_set_works_before_and_after_the_subcommand():
    """Both orders must work: users naturally append flags at the end."""
    from trace_rag.cli import build_parser

    parser = build_parser()
    before = parser.parse_args(["--set", "generation.backend=stub", "query", "hi"])
    after = parser.parse_args(["query", "hi", "--set", "generation.backend=stub"])
    assert before.set == ["generation.backend=stub"]
    assert after.set == ["generation.backend=stub"]
    mixed = parser.parse_args(["--root", "r", "query", "hi", "--set", "retrieval.top_k=3"])
    assert mixed.root == "r" and mixed.set == ["retrieval.top_k=3"]


def test_cli_query_with_trailing_overrides(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    root = str(tmp_path / "run")
    assert main(["--root", root, "ingest-beir", "--corpus", str(corpus), "--n-sources", "2"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "index"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "query", "Who designed the Eiffel Tower?", "--json",
                 "--set", "generation.backend=stub"]) == 0
    assert json.loads(capsys.readouterr().out)["answer"]["query"]


def test_ingest_does_not_build_the_embedder(tmp_path, monkeypatch, capsys):
    """Ingestion must not load the embedding model (it would download Contriever)."""
    import trace_rag.embeddings as embeddings

    called = {"n": 0}
    original = embeddings.build_embedder

    def counting(config):
        called["n"] += 1
        return original(config)

    monkeypatch.setattr(embeddings, "build_embedder", counting)
    monkeypatch.setattr("trace_rag.pipeline.build_embedder", counting)

    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    assert main(["--root", str(tmp_path / "run"), "ingest-beir", "--corpus", str(corpus)]) == 0
    capsys.readouterr()
    assert called["n"] == 0, "ingestion loaded the embedding model"


def test_remediate_does_not_build_the_embedder(tmp_path, monkeypatch, capsys):
    called = {"n": 0}
    monkeypatch.setattr("trace_rag.pipeline.build_embedder",
                        lambda config: called.__setitem__("n", called["n"] + 1))
    assert main(["--root", str(tmp_path / "run"), "remediate", "--doc-ids", "x#0000"]) == 0
    capsys.readouterr()
    assert called["n"] == 0
