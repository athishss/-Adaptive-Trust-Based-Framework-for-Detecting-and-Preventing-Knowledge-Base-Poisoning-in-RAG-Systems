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
