"""NELA-GT loader: real outlets and times as provenance (format fixture, made-up values)."""
from __future__ import annotations

import pytest


def _nela_fixture(tmp_path):
    """Matches the field list in the NELA-GT-2020 paper (Table 1); the values are made up."""
    import sqlite3

    db = tmp_path / "nela.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE newsdata (id TEXT, date TEXT, source TEXT, title TEXT, content TEXT, "
                "author TEXT, published TEXT, published_utc INTEGER, collection_utc INTEGER, url TEXT)")
    body = "word " * 80
    rows = [("a1", "2020-03-01", "Outlet A", "T1", body, "x", "Mar 1", 1583020800, 1583020900, "http://a/1"),
            ("a2", "2020-03-02", "Outlet A", "T2", body, "x", "Mar 2", 1583107200, 1583107300, "http://a/2"),
            ("b1", "2020-03-03", "Outlet B", "T3", body, "y", "Mar 3", 1583193600, 1583193700, "http://b/1"),
            ("c1", "2020-03-04", "Outlet C", "short", "tiny", "z", "Mar 4", 1583280000, 1583280100, "http://c/1")]
    con.executemany("INSERT INTO newsdata VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    labels = tmp_path / "labels.csv"
    labels.write_text("source,label\nOutlet A,reliable\nOutlet B,unreliable\n", encoding="utf-8")
    return db, labels


def test_nela_ingests_real_outlets_and_times(tmp_path):
    from trace_rag.config import Config
    from trace_rag.ingestion import Ingestor, ProvenanceStore, ingest_nela, load_source_labels

    db, labels_csv = _nela_fixture(tmp_path)
    labels = load_source_labels(labels_csv)
    assert labels == {"Outlet A": "reliable", "Outlet B": "unreliable"}
    cfg = Config.load(None, storage={"root": str(tmp_path / "run")})
    with ProvenanceStore(tmp_path / "p.sqlite3") as store:
        report = ingest_nela(Ingestor(store, cfg.ingestion), db, labels)
        assert report["documents"] == 3 and report["skipped"] == 1       # the too-short article is skipped
        assert report["by_label"] == {"reliable": 2, "unreliable": 1}
        assert report["sources"] == 2
        assert store.counts()["sources"] == 2                            # real outlets, not simulated ids


def test_nela_refuses_ambiguous_integer_labels(tmp_path):
    from trace_rag.ingestion import load_source_labels

    path = tmp_path / "l.csv"
    path.write_text("source,label\nOutlet A,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="int_map"):
        load_source_labels(path)
    assert load_source_labels(path, {0: "reliable"}) == {"Outlet A": "reliable"}
