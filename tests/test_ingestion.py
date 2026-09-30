from __future__ import annotations

import time

import pytest

from trace_rag.config import IngestionConfig
from trace_rag.ingestion import (Ingestor, ParseError, SourceAssigner,
                                 chunk_passage, chunk_text, parse_file)
from trace_rag.utils.hashing import FamilyAssigner


def test_chunking_covers_all_words_with_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), chunk_words=100, overlap_words=20)
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert chunks[0].end_word == 100 and chunks[1].start_word == 80      # overlap respected
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start_word, chunk.end_word))
    assert covered == set(range(250))


def test_chunking_is_deterministic():
    text = " ".join(f"w{i}" for i in range(500))
    assert [c.text for c in chunk_text(text)] == [c.text for c in chunk_text(text)]


def test_chunking_merges_short_tail():
    words = " ".join(f"w{i}" for i in range(105))
    chunks = chunk_text(words, chunk_words=100, overlap_words=20, min_words=15)
    assert all(c.n_words >= 15 for c in chunks)


def test_chunking_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=10, overlap_words=10)


def test_passage_mode_keeps_short_passages_whole():
    assert len(chunk_passage("a short benchmark passage", chunk_words=100)) == 1


def test_parse_txt_md_html(tmp_path):
    (tmp_path / "a.txt").write_text("plain   text\nhere", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nSome **bold** [link](http://x) text", encoding="utf-8")
    (tmp_path / "c.html").write_text("<html><body><script>x=1</script><p>Hello world</p></body></html>",
                                     encoding="utf-8")
    assert parse_file(tmp_path / "a.txt") == "plain text here"
    assert "bold" in parse_file(tmp_path / "b.md") and "http" not in parse_file(tmp_path / "b.md")
    html_text = parse_file(tmp_path / "c.html")
    assert "Hello world" in html_text and "x=1" not in html_text


def test_parse_unsupported_type_raises(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "x.bin")


def test_provenance_records_full_chain(store):
    ingestor = Ingestor(store, IngestionConfig())
    records = ingestor.ingest_text("d1", "word " * 300, "src_a", ingested_at=1000.0)
    assert records and all(r.source_id == "src_a" and r.family_id and r.sha256 for r in records)
    assert store.counts() == {"sources": 1, "documents": 1, "chunks": len(records)}
    assert store.get_chunk(records[0].chunk_id).doc_id == "d1"


def test_reingesting_identical_content_is_a_noop(store):
    ingestor = Ingestor(store, IngestionConfig())
    ingestor.ingest_text("d1", "same text here", "src", ingested_at=1.0, passage_mode=True)
    before = store.counts()
    ingestor.ingest_text("d1", "same text here", "src", ingested_at=2.0, passage_mode=True)
    assert store.counts() == before


def test_reingesting_changed_content_bumps_version(store):
    ingestor = Ingestor(store, IngestionConfig())
    ingestor.ingest_text("d1", "first version of the text", "src", ingested_at=1.0, passage_mode=True)
    ingestor.ingest_text("d1", "second and different version", "src", ingested_at=2.0, passage_mode=True)
    assert store.counts()["documents"] == 1
    assert store.get_chunk("d1#0000").text == "second and different version"


def test_burst_count_window(store):
    ingestor = Ingestor(store, IngestionConfig())
    for i in range(5):
        ingestor.ingest_text(f"burst{i}", f"passage number {i} from the attacker", "attacker",
                             ingested_at=1_000_000.0 + i * 60, passage_mode=True)
    ingestor.ingest_text("old", "an older passage", "attacker", ingested_at=1.0, passage_mode=True)
    assert store.burst_count("attacker", 1_000_000.0, window_hours=1.0) == 5
    assert store.burst_count("attacker", 1.0, window_hours=1.0) == 1


def test_source_stats_age(store):
    ingestor = Ingestor(store, IngestionConfig())
    now = time.time()
    ingestor.ingest_text("d", "text", "src", ingested_at=now - 10 * 86400, passage_mode=True)
    stats = store.source_stats("src")
    assert 9.5 < stats.age_days(now) < 10.5 and stats.n_docs == 1


def test_near_duplicate_passages_share_a_family():
    assigner = FamilyAssigner()
    base = "The Eiffel Tower was completed in Paris in 1889 for the World Fair exhibition"
    near = "The Eiffel Tower was completed in Paris in 1889 for the World Fair exposition"
    far = "Photosynthesis converts light energy into chemical energy in plant cells"
    fam_a = assigner.assign("c1", base)
    fam_b = assigner.assign("c2", near)
    fam_c = assigner.assign("c3", far)
    assert fam_a == fam_b != fam_c
    assert assigner.family_size(fam_a) == 2


def test_exact_duplicates_always_share_a_family():
    assigner = FamilyAssigner()
    text = "identical passage text used twice"
    assert assigner.assign("a", text) == assigner.assign("b", text)


def test_source_assigner_is_deterministic_and_spread():
    a = SourceAssigner(n_sources=100, seed=7)
    b = SourceAssigner(n_sources=100, seed=7)
    assert [a.assign(f"d{i}") for i in range(50)] == [b.assign(f"d{i}") for i in range(50)]
    assigned = {a.assign(f"d{i}") for i in range(500)}
    assert 10 < len(assigned) <= 100


def test_beir_ingestion_spreads_timestamps(tmp_path, store):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(
        '{"_id": "c%d", "title": "t%d", "text": "passage body number %d about topics"}' % (i, i, i)
        for i in range(5)), encoding="utf-8")
    ingestor = Ingestor(store, IngestionConfig())
    report = ingestor.ingest_beir(corpus, SourceAssigner(n_sources=3), start_time=1000.0,
                                  seconds_per_doc=3600.0)
    assert report.documents == 5
    times = sorted({c.ingested_at for c in store.iter_chunks()})
    assert len(times) == 5 and times[-1] - times[0] == 4 * 3600.0


def test_malformed_beir_line_raises(tmp_path, store):
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text('{"_id": "ok", "text": "fine"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ParseError):
        Ingestor(store, IngestionConfig()).ingest_beir(corpus, SourceAssigner(n_sources=2))
