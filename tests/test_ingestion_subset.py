from __future__ import annotations

import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from trace_rag.ingestion.subset import (  # noqa: E402
    build_exact_subset,
    read_qrels_set,
    select_query_ids,
    verify_subset,
    write_queries_exact,
)


def _write_corpus(path, ids, *, title_prefix="t", text_prefix="passage"):
    pq.write_table(pa.table({
        "_id": pa.array(ids),
        "title": pa.array([f"{title_prefix}{i}" for i in range(len(ids))]),
        "text": pa.array([f"{text_prefix} {i}" for i in range(len(ids))]),
    }), path)
    return path


def _write_queries(path, rows):
    pq.write_table(pa.table({
        "_id": pa.array([row[0] for row in rows]),
        "text": pa.array([row[1] for row in rows]),
    }), path)
    return path


def _read_ids(path):
    return pq.read_table(path, columns=["_id"]).column("_id").to_pylist()


def _write_qrels(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_read_qrels_accepts_header_and_no_header(tmp_path):
    trend = ("query-id\tcorpus-id\tscore\n"
             "q0\td0\t1\nq0\td1\t0\nq1\td2\t2\n")
    no_header = "q0\td0\t1\nq0\td1\t0\nq1\td2\t2\n"
    expected = {"q0": {"d0"}, "q1": {"d2"}}
    assert read_qrels_set(_write_qrels(tmp_path / "header.tsv", trend)) == expected
    assert read_qrels_set(_write_qrels(tmp_path / "no_header.tsv", no_header)) == expected


def test_read_qrels_ignores_blank_lines_and_bom(tmp_path):
    path = _write_qrels(
        tmp_path / "qrels.tsv",
        "\ufeffquery-id\tcorpus-id\tscore\n\nq0\td0\t1\n\nq1\td1\t3\n")
    assert read_qrels_set(path) == {"q0": {"d0"}, "q1": {"d1"}}


@pytest.mark.parametrize("body", [
    "q0\td0\n",
    "q0\td0\tnot-a-number\n",
    "q0\t\t1\n",
    "\t d0\t1\n",
    "q0\td0\tinf\n",
])
def test_read_qrels_rejects_malformed_rows(tmp_path, body):
    with pytest.raises(ValueError):
        read_qrels_set(_write_qrels(tmp_path / "bad.tsv", body))


def test_build_exact_subset_has_exact_size_unique_ids_and_full_gold(tmp_path):
    ids = [f"d{i}" for i in range(2_000)]
    gold = {f"d{i}" for i in range(0, 2_000, 97)}
    corpus = _write_corpus(tmp_path / "corpus.parquet", ids)
    out = tmp_path / "subset.parquet"

    kept, gold_kept = build_exact_subset(corpus, gold, 300, out)

    assert kept == 300
    assert gold_kept == len(gold)
    selected = _read_ids(out)
    assert len(selected) == 300
    assert len(set(selected)) == 300
    assert gold <= set(selected)
    # Original corpus order is retained, so provenance is easy to audit.
    assert selected == [doc_id for doc_id in ids if doc_id in set(selected)]


def test_build_exact_subset_is_deterministic_for_the_same_seed(tmp_path):
    ids = [f"d{i}" for i in range(500)]
    corpus = _write_corpus(tmp_path / "corpus.parquet", ids)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    build_exact_subset(corpus, {"d1", "d400"}, 100, first)
    build_exact_subset(corpus, {"d1", "d400"}, 100, second)

    assert _read_ids(first) == _read_ids(second)
    assert first.read_bytes() == second.read_bytes()


def test_build_exact_subset_changes_sampled_rows_with_the_seed(tmp_path):
    ids = [f"d{i}" for i in range(100)]
    corpus = _write_corpus(tmp_path / "corpus.parquet", ids)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    build_exact_subset(corpus, {"d0"}, 10, first, seed=1)
    build_exact_subset(corpus, {"d0"}, 10, second, seed=2)

    assert set(_read_ids(first)) != set(_read_ids(second))


def test_build_exact_subset_supports_an_all_gold_corpus(tmp_path):
    ids = [f"d{i}" for i in range(10)]
    corpus = _write_corpus(tmp_path / "corpus.parquet", ids)
    out = tmp_path / "subset.parquet"

    kept, gold_kept = build_exact_subset(corpus, set(ids), 10, out)

    assert (kept, gold_kept) == (10, 10)
    assert _read_ids(out) == ids


def test_build_exact_subset_rejects_missing_gold_without_output(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.parquet", [f"d{i}" for i in range(50)])
    out = tmp_path / "subset.parquet"

    with pytest.raises(ValueError, match="absent from the corpus"):
        build_exact_subset(corpus, {"d1", "missing-gold"}, 10, out)

    assert not out.exists()
    assert list(tmp_path.glob("subset.parquet.*.tmp")) == []


def test_build_exact_subset_rejects_insufficient_rows_without_output(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.parquet", [f"d{i}" for i in range(5)])
    out = tmp_path / "subset.parquet"

    with pytest.raises(ValueError, match="only 5 rows"):
        build_exact_subset(corpus, {"d0"}, 10, out)

    assert not out.exists()


def test_build_exact_subset_rejects_target_smaller_than_gold(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.parquet", [f"d{i}" for i in range(10)])

    with pytest.raises(ValueError, match="smaller than"):
        build_exact_subset(corpus, {"d0", "d1", "d2"}, 2, tmp_path / "out.parquet")


def test_build_exact_subset_rejects_duplicate_gold_ids(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.parquet", ["d0", "d0", "d1", "d2"])

    with pytest.raises(ValueError, match="duplicate gold document id"):
        build_exact_subset(corpus, {"d0"}, 3, tmp_path / "out.parquet")


def test_build_exact_subset_rejects_duplicate_sampled_non_gold_ids(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.parquet", ["d0", "x", "x"])

    with pytest.raises(ValueError, match="duplicate selected document ids"):
        build_exact_subset(corpus, {"d0"}, 3, tmp_path / "out.parquet")


def test_build_exact_subset_rejects_missing_columns(tmp_path):
    pq.write_table(pa.table({"_id": pa.array(["d0"]), "text": pa.array(["p"])}),
                   tmp_path / "corpus.parquet")

    with pytest.raises(ValueError, match="missing required columns"):
        build_exact_subset(tmp_path / "corpus.parquet", {"d0"}, 1, tmp_path / "out.parquet")


def test_write_queries_exact_writes_requested_ids_in_order(tmp_path):
    queries = _write_queries(tmp_path / "queries.parquet",
                             [(f"q{i}", f"question {i}") for i in range(5)])
    out = tmp_path / "queries.jsonl"

    written = write_queries_exact(queries, ["q3", "q0"], out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == 2
    assert [row["qid"] for row in rows] == ["q3", "q0"]
    assert rows == [{"qid": "q3", "text": "question 3"},
                    {"qid": "q0", "text": "question 0"}]


def test_write_queries_exact_rejects_missing_query_ids(tmp_path):
    queries = _write_queries(tmp_path / "queries.parquet", [("q0", "question 0")])
    out = tmp_path / "queries.jsonl"

    with pytest.raises(ValueError, match="absent from the queries file"):
        write_queries_exact(queries, ["q0", "q99"], out)

    assert not out.exists()


@pytest.mark.parametrize("selected", [[], ["q0", "q0"], ["q0", " "], [None]])
def test_write_queries_exact_rejects_empty_duplicate_or_invalid_ids(tmp_path, selected):
    queries = _write_queries(tmp_path / "queries.parquet", [("q0", "question 0")])

    with pytest.raises(ValueError):
        write_queries_exact(queries, selected, tmp_path / "queries.jsonl")


@pytest.mark.parametrize("text", ["", "   ", None])
def test_write_queries_exact_rejects_queries_without_text(tmp_path, text):
    queries = _write_queries(tmp_path / "queries.parquet", [("q0", text)])

    with pytest.raises(ValueError, match="no non-empty text"):
        write_queries_exact(queries, ["q0"], tmp_path / "queries.jsonl")


def test_write_queries_exact_rejects_duplicate_source_qids(tmp_path):
    queries = _write_queries(tmp_path / "queries.parquet",
                             [("q0", "first"), ("q0", "second")])

    with pytest.raises(ValueError, match="duplicate query id"):
        write_queries_exact(queries, ["q0"], tmp_path / "queries.jsonl")


def _prepared_contract(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.parquet", ["d0", "d1", "d2", "d3"])
    queries = _write_queries(tmp_path / "queries.parquet",
                             [("q0", "question 0"), ("q1", "question 1")])
    qrels = _write_qrels(tmp_path / "qrels.tsv",
                         "query-id\tcorpus-id\tscore\nq0\td0\t1\nq1\td2\t2\n")
    subset = tmp_path / "subset.parquet"
    build_exact_subset(corpus, {"d0", "d2"}, 3, subset, seed=7)
    query_subset = tmp_path / "queries.jsonl"
    write_queries_exact(queries, ["q0", "q1"], query_subset)
    return subset, query_subset, qrels


def test_verify_subset_accepts_an_exact_contract(tmp_path):
    subset, queries, qrels = _prepared_contract(tmp_path)

    report = verify_subset(subset, queries, qrels, 3, 2)

    assert report["ok"] is True, report["errors"]
    assert report["corpus_rows"] == 3
    assert report["query_rows"] == 2
    assert report["gold_passages_required"] == 2
    assert report["gold_passages_present"] == 2
    assert report["gold_coverage"] == 1.0


def test_verify_subset_detects_duplicate_corpus_ids(tmp_path):
    corpus = _write_corpus(tmp_path / "subset.parquet", ["d0", "d0", "d1"])
    queries = _write_queries(tmp_path / "queries.parquet", [("q0", "question 0")])
    query_subset = tmp_path / "queries.jsonl"
    write_queries_exact(queries, ["q0"], query_subset)
    qrels = _write_qrels(tmp_path / "qrels.tsv", "q0\td0\t1\n")

    report = verify_subset(corpus, query_subset, qrels, 3, 1)

    assert report["ok"] is False
    assert report["duplicate_corpus_ids"] == 1
    assert any("duplicate ids" in error for error in report["errors"])


def test_verify_subset_detects_missing_query_gold_and_missing_gold_passages(tmp_path):
    corpus = _write_corpus(tmp_path / "subset.parquet", ["d0"])
    queries = _write_queries(tmp_path / "queries.parquet",
                             [("q0", "question 0"), ("q1", "question 1")])
    query_subset = tmp_path / "queries.jsonl"
    write_queries_exact(queries, ["q0", "q1"], query_subset)
    qrels = _write_qrels(tmp_path / "qrels.tsv", "q0\td9\t1\n")

    report = verify_subset(corpus, query_subset, qrels, 1, 2)

    assert report["ok"] is False
    assert report["missing_gold"] == ["d9"]
    assert report["missing_query_gold"] == ["q1"]
    assert report["gold_coverage"] == 0.0
    assert any("required gold passages are absent" in error for error in report["errors"])
    assert any("no positive qrels" in error for error in report["errors"])


def test_verify_subset_detects_wrong_expected_counts(tmp_path):
    subset, queries, qrels = _prepared_contract(tmp_path)

    report = verify_subset(subset, queries, qrels, 4, 3)

    assert report["ok"] is False
    assert any("expected exactly 4" in error for error in report["errors"])
    assert any("expected 3" in error for error in report["errors"])


def test_verify_subset_reports_invalid_query_lines_instead_of_crashing(tmp_path):
    corpus = _write_corpus(tmp_path / "subset.parquet", ["d0"])
    query_subset = tmp_path / "queries.jsonl"
    query_subset.write_text(
        '{"qid": "q0", "text": "question 0"}\n'
        'not json\n'
        '{"text": "missing qid"}\n'
        '{"qid": "q1", "text": "  "}\n'
        '{"qid": "q0", "text": "duplicate"}\n',
        encoding="utf-8")
    qrels = _write_qrels(tmp_path / "qrels.tsv", "q0\td0\t1\n")

    report = verify_subset(corpus, query_subset, qrels, 1, 4)

    combined = " | ".join(report["errors"])
    assert report["ok"] is False
    assert "not valid JSON" in combined
    assert "no non-empty qid" in combined
    assert "no non-empty text" in combined
    assert "duplicate qid" in combined


def test_select_query_ids_is_deterministic_exact_and_seed_sensitive():
    gold = {f"q{i}": {f"d{i}"} for i in range(50)}

    first = select_query_ids(gold, 10, seed=11)
    again = select_query_ids(gold, 10, seed=11)
    other = select_query_ids(gold, 10, seed=12)

    assert first == again
    assert len(first) == 10 and len(set(first)) == 10
    assert set(first) <= set(gold)
    assert first != other


def test_select_query_ids_rejects_empty_or_invalid_qrels():
    with pytest.raises(ValueError, match="only 0"):
        select_query_ids({}, 1)
    with pytest.raises(ValueError, match="non-empty identifier"):
        select_query_ids({"": {"d0"}}, 1)
    with pytest.raises(TypeError):
        select_query_ids({"q0": {"d0"}}, True)  # type: ignore[arg-type]
