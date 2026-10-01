"""Provenance store: sources, documents, chunks and their trust-relevant history.

This is Person A's write-side store.  Person B's trust ledger is a *separate*
database; this one records who supplied what and when, which is exactly the
information the cheap signals S4 (ingestion burst) and S5 (source maturity)
need, and which Person B's cold-start priors read.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from ..utils.hashing import sha256_text

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    n_docs      INTEGER NOT NULL DEFAULT 0,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    sha256       TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    origin       TEXT NOT NULL DEFAULT '',
    version      INTEGER NOT NULL DEFAULT 1,
    ingested_at  REAL NOT NULL,
    n_chunks     INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    source_id   TEXT NOT NULL,
    family_id   TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    n_words     INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    ingested_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc     ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source  ON chunks(source_id, ingested_at);
CREATE INDEX IF NOT EXISTS idx_chunks_family  ON chunks(family_id);
CREATE INDEX IF NOT EXISTS idx_docs_source    ON documents(source_id, ingested_at);
"""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_id: str
    family_id: str
    ordinal: int
    text: str
    n_words: int
    sha256: str
    ingested_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "source_id": self.source_id,
            "family_id": self.family_id, "ordinal": self.ordinal, "text": self.text,
            "n_words": self.n_words, "sha256": self.sha256, "ingested_at": self.ingested_at,
        }


@dataclass(frozen=True)
class SourceStats:
    source_id: str
    first_seen: float
    last_seen: float
    n_docs: int
    n_chunks: int

    def age_days(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, (now - self.first_seen) / 86400.0)


class ProvenanceStore:
    """Thread-safe SQLite store.  All timestamps are POSIX seconds (UTC)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._deferred = 0                      # >0 while inside batch(): commits are deferred
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        except sqlite3.OperationalError as exc:
            raise sqlite3.OperationalError(
                f"cannot open the provenance store at {self.path}: {exc}. Check that the folder exists "
                f"and is writable (storage.root in your config)."
            ) from exc
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def _commit(self) -> None:
        if self._deferred == 0:
            self._conn.commit()

    @contextmanager
    def batch(self):
        """Defer commits for a bulk load.

        One commit per document turns a 2.68M-passage ingest into millions of
        fsyncs.  Inside this block the writes share a single transaction, which
        is roughly an order of magnitude faster; the block commits on exit and
        rolls back if the body raises.
        """
        with self._lock:
            self._deferred += 1
        try:
            yield self
        except Exception:
            with self._lock:
                self._deferred -= 1
                if self._deferred == 0:
                    self._conn.rollback()
            raise
        else:
            with self._lock:
                self._deferred -= 1
                if self._deferred == 0:
                    self._conn.commit()

    # ------------------------------------------------------------------ write
    def upsert_source(self, source_id: str, seen_at: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        seen = time.time() if seen_at is None else float(seen_at)
        with self._lock:
            self._conn.execute(
                """INSERT INTO sources(source_id, first_seen, last_seen, metadata)
                   VALUES(?, ?, ?, ?)
                   ON CONFLICT(source_id) DO UPDATE SET
                       last_seen = MAX(sources.last_seen, excluded.last_seen),
                       first_seen = MIN(sources.first_seen, excluded.first_seen)""",
                (source_id, seen, seen, json.dumps(metadata or {}, sort_keys=True)),
            )
            self._commit()

    def add_document(self, doc_id: str, source_id: str, text_sha256: str, chunks: Sequence[ChunkRecord],
                     title: str = "", origin: str = "", ingested_at: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert a document and its chunks.  Re-ingesting the same doc_id with
        different content bumps ``version`` and replaces its chunks."""
        ts = time.time() if ingested_at is None else float(ingested_at)
        with self._lock:
            self.upsert_source(source_id, ts)
            row = self._conn.execute(
                "SELECT sha256, version, source_id, n_chunks FROM documents WHERE doc_id = ?",
                (doc_id,)
            ).fetchone()
            version = 1
            old_source: Optional[str] = None
            old_chunks = 0
            if row is not None:
                if row["sha256"] == text_sha256:
                    return int(row["version"])          # identical re-ingest: no-op
                version = int(row["version"]) + 1
                old_source = row["source_id"]
                old_chunks = int(row["n_chunks"])
                self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._conn.execute(
                """INSERT INTO documents(doc_id, source_id, sha256, title, origin, version,
                                         ingested_at, n_chunks, metadata)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (doc_id, source_id, text_sha256, title, origin, version, ts, len(chunks),
                 json.dumps(metadata or {}, sort_keys=True)),
            )
            self._conn.executemany(
                """INSERT OR REPLACE INTO chunks(chunk_id, doc_id, source_id, family_id, ordinal,
                                                 text, n_words, sha256, ingested_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                [(c.chunk_id, c.doc_id, c.source_id, c.family_id, c.ordinal, c.text,
                  c.n_words, c.sha256, c.ingested_at) for c in chunks],
            )
            # Counters are maintained incrementally.  Recomputing them with COUNT(*)
            # per document made ingestion quadratic (minutes per 10k passages).
            if old_source is not None and old_source != source_id:
                self._conn.execute(
                    "UPDATE sources SET n_docs = MAX(0, n_docs - 1), "
                    "n_chunks = MAX(0, n_chunks - ?) WHERE source_id = ?",
                    (old_chunks, old_source),
                )
                delta_docs, delta_chunks = 1, len(chunks)
            elif old_source is not None:
                delta_docs, delta_chunks = 0, len(chunks) - old_chunks
            else:
                delta_docs, delta_chunks = 1, len(chunks)
            self._conn.execute(
                "UPDATE sources SET n_docs = MAX(0, n_docs + ?), n_chunks = MAX(0, n_chunks + ?), "
                "last_seen = MAX(last_seen, ?) WHERE source_id = ?",
                (delta_docs, delta_chunks, ts, source_id),
            )
            self._commit()
        return version

    # ------------------------------------------------------------------- read
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        out: Dict[str, ChunkRecord] = {}
        with self._lock:
            for batch_start in range(0, len(chunk_ids), 500):
                batch = list(chunk_ids[batch_start:batch_start + 500])
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", batch
                ).fetchall()
                for row in rows:
                    out[row["chunk_id"]] = self._row_to_chunk(row)
        return out

    def iter_chunks(self, batch_size: int = 1000) -> Iterable[ChunkRecord]:
        """Stream every chunk in chunk_id order.

        Keyset pagination ("where chunk_id > last"), not OFFSET: OFFSET makes
        SQLite re-scan and discard every earlier row on each page, so iteration
        slows down the further it gets - measured 0.3s for the first 50k rows
        and 0.7s for the fourth 50k, and it keeps growing with corpus size.
        """
        last_id = ""
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM chunks WHERE chunk_id > ? ORDER BY chunk_id LIMIT ?",
                    (last_id, batch_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._row_to_chunk(row)
            last_id = rows[-1]["chunk_id"]

    def source_stats(self, source_id: str) -> Optional[SourceStats]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        if row is None:
            return None
        return SourceStats(row["source_id"], row["first_seen"], row["last_seen"],
                           int(row["n_docs"]), int(row["n_chunks"]))

    def burst_count(self, source_id: str, around: float, window_hours: float = 24.0) -> int:
        """Chunks from ``source_id`` ingested within +/- window of ``around``.

        This is the raw quantity behind signal S4.  It counts chunks, not
        documents, because an attacker can pack many passages into one upload.
        """
        half = float(window_hours) * 3600.0
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE source_id = ? AND ingested_at BETWEEN ? AND ?",
                (source_id, around - half, around + half),
            ).fetchone()
        return int(row["n"])

    def family_size(self, family_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE family_id = ?", (family_id,)
            ).fetchone()
        return int(row["n"])

    def family_sizes(self, family_ids: Sequence[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        uniq = sorted(set(family_ids))
        placeholders = ",".join("?" * len(uniq))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT family_id, COUNT(*) AS n FROM chunks WHERE family_id IN ({placeholders}) "
                f"GROUP BY family_id", uniq
            ).fetchall()
        return {row["family_id"]: int(row["n"]) for row in rows}

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "sources": int(self._conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"]),
                "documents": int(self._conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]),
                "chunks": int(self._conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]),
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ProvenanceStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], source_id=row["source_id"],
            family_id=row["family_id"], ordinal=int(row["ordinal"]), text=row["text"],
            n_words=int(row["n_words"]), sha256=row["sha256"], ingested_at=float(row["ingested_at"]),
        )


_SAFE_EXTRA = "-_."
_MAX_PLAIN = 48
_PREFIX = 40


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    """Stable, collision-free chunk id.

    Benchmark ids (``nq_1``) are used as they are, so ids stay readable.  Any id
    that needs sanitising or is longer than 48 characters gets a hash of the
    *full* original appended after a ``~``.  ``~`` is never produced by the
    sanitiser, so a hashed id can never equal a plain one.

    Without that hash, ``doc/one`` and ``doc one`` (or two long ids sharing a
    prefix) mapped to the same chunk id and silently overwrote each other's
    rows in the store.
    """
    safe = "".join(ch if ch.isalnum() or ch in _SAFE_EXTRA else "_" for ch in doc_id)
    if safe == doc_id and len(doc_id) <= _MAX_PLAIN:
        return f"{safe}#{ordinal:04d}"
    digest = sha256_text(doc_id)[:12]
    return f"{safe[:_PREFIX]}~{digest}#{ordinal:04d}"
