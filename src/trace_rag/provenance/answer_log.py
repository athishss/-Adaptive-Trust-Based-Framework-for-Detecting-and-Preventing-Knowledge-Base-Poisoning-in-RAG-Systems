"""Answer-provenance log: which passages produced which answer, and when.

This is what makes retroactive remediation possible (plan Section 4.8).  Every
answer is written with the passages it cited *and* the trust values at the time,
so when Person B's ledger later quarantines a passage we can say exactly which
past answers are affected.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import AnswerRecord

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS answers (
    answer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    TEXT NOT NULL,
    step        INTEGER NOT NULL DEFAULT 0,
    timestamp   REAL NOT NULL,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    abstained   INTEGER NOT NULL,
    abstain_reason TEXT,
    evidence_mass  REAL NOT NULL DEFAULT 0,
    llm_calls   INTEGER NOT NULL DEFAULT 0,
    latency_ms  REAL NOT NULL DEFAULT 0,
    model       TEXT NOT NULL DEFAULT '',
    flagged     INTEGER NOT NULL DEFAULT 0,
    flag_reason TEXT,
    flagged_at  REAL,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS answer_documents (
    answer_id   INTEGER NOT NULL REFERENCES answers(answer_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    role        TEXT NOT NULL,           -- 'cited' | 'context' | 'excluded'
    t_eff       REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (answer_id, doc_id, role)
);

CREATE INDEX IF NOT EXISTS idx_answer_docs_doc ON answer_documents(doc_id, role);
CREATE INDEX IF NOT EXISTS idx_answers_step    ON answers(step);
CREATE INDEX IF NOT EXISTS idx_answers_flagged ON answers(flagged);
"""


@dataclass(frozen=True)
class StoredAnswer:
    answer_id: int
    query_id: str
    step: int
    timestamp: float
    query: str
    answer: str
    abstained: bool
    abstain_reason: Optional[str]
    evidence_mass: float
    llm_calls: int
    latency_ms: float
    model: str
    flagged: bool
    flag_reason: Optional[str]
    flagged_at: Optional[float]
    cited_doc_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id, "query_id": self.query_id, "step": self.step,
            "timestamp": self.timestamp, "query": self.query, "answer": self.answer,
            "abstained": self.abstained, "abstain_reason": self.abstain_reason,
            "evidence_mass": self.evidence_mass, "llm_calls": self.llm_calls,
            "latency_ms": self.latency_ms, "model": self.model, "flagged": self.flagged,
            "flag_reason": self.flag_reason, "flagged_at": self.flagged_at,
            "cited_doc_ids": list(self.cited_doc_ids),
        }


class AnswerLog:
    """Thread-safe SQLite answer log."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record(self, record: AnswerRecord, context_doc_ids: Sequence[str] = ()) -> int:
        payload = json.dumps(record.to_dict(), sort_keys=True)
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO answers(query_id, step, timestamp, query, answer, abstained,
                                       abstain_reason, evidence_mass, llm_calls, latency_ms,
                                       model, payload)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record.query_id, int(record.step), float(record.timestamp), record.query,
                 record.answer, int(record.abstained), record.abstain_reason,
                 float(record.evidence_mass), int(record.llm_calls), float(record.latency_ms),
                 record.model, payload),
            )
            answer_id = int(cursor.lastrowid)
            rows: List[Tuple[int, str, str, float]] = []
            trust = record.trust_snapshots
            for doc_id in record.used_doc_ids:
                rows.append((answer_id, doc_id, "cited", float(trust.get(doc_id, {}).get("t_eff", 0.0))))
            for doc_id in context_doc_ids:
                rows.append((answer_id, doc_id, "context", float(trust.get(doc_id, {}).get("t_eff", 0.0))))
            for doc_id in record.excluded_doc_ids:
                rows.append((answer_id, doc_id, "excluded", float(trust.get(doc_id, {}).get("t_eff", 0.0))))
            self._conn.executemany(
                "INSERT OR REPLACE INTO answer_documents(answer_id, doc_id, role, t_eff) VALUES(?,?,?,?)",
                rows,
            )
            self._conn.commit()
        return answer_id

    def answers_using(self, doc_ids: Sequence[str], roles: Sequence[str] = ("cited", "context"),
                      only_unflagged: bool = False) -> List[StoredAnswer]:
        if not doc_ids:
            return []
        doc_ids = list(dict.fromkeys(doc_ids))
        roles = list(roles)
        out: Dict[int, StoredAnswer] = {}
        with self._lock:
            for start in range(0, len(doc_ids), 400):
                batch = doc_ids[start:start + 400]
                doc_ph = ",".join("?" * len(batch))
                role_ph = ",".join("?" * len(roles))
                query = (
                    f"SELECT DISTINCT a.* FROM answers a "
                    f"JOIN answer_documents d ON d.answer_id = a.answer_id "
                    f"WHERE d.doc_id IN ({doc_ph}) AND d.role IN ({role_ph})"
                )
                params: List[Any] = batch + roles
                if only_unflagged:
                    query += " AND a.flagged = 0"
                for row in self._conn.execute(query, params).fetchall():
                    out[int(row["answer_id"])] = self._row_to_answer(row)
        return [out[k] for k in sorted(out)]

    def flag(self, answer_ids: Sequence[int], reason: str, at: Optional[float] = None) -> int:
        if not answer_ids:
            return 0
        stamp = time.time() if at is None else float(at)
        with self._lock:
            cursor = self._conn.executemany(
                "UPDATE answers SET flagged = 1, flag_reason = ?, flagged_at = ? "
                "WHERE answer_id = ? AND flagged = 0",
                [(reason, stamp, int(a)) for a in answer_ids],
            )
            self._conn.commit()
            changed = int(cursor.rowcount if cursor.rowcount is not None else 0)
        return max(0, changed)

    def get(self, answer_id: int) -> Optional[StoredAnswer]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM answers WHERE answer_id = ?", (answer_id,)).fetchone()
        return self._row_to_answer(row) if row else None

    def payload(self, answer_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM answers WHERE answer_id = ?",
                                     (answer_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def all_answers(self, limit: Optional[int] = None) -> List[StoredAnswer]:
        sql = "SELECT * FROM answers ORDER BY answer_id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [self._row_to_answer(row) for row in rows]

    def stats(self) -> Dict[str, float]:
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) n, SUM(abstained) abstained, SUM(flagged) flagged,
                          AVG(llm_calls) avg_calls, AVG(latency_ms) avg_latency
                   FROM answers"""
            ).fetchone()
        total = int(row["n"] or 0)
        return {
            "answers": total,
            "abstained": int(row["abstained"] or 0),
            "flagged": int(row["flagged"] or 0),
            "abstention_rate": (int(row["abstained"] or 0) / total) if total else 0.0,
            "mean_llm_calls": float(row["avg_calls"] or 0.0),
            "mean_latency_ms": float(row["avg_latency"] or 0.0),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "AnswerLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _row_to_answer(self, row: sqlite3.Row) -> StoredAnswer:
        with self._lock:
            cited = [r["doc_id"] for r in self._conn.execute(
                "SELECT doc_id FROM answer_documents WHERE answer_id = ? AND role = 'cited' "
                "ORDER BY doc_id", (row["answer_id"],)).fetchall()]
        return StoredAnswer(
            answer_id=int(row["answer_id"]), query_id=row["query_id"], step=int(row["step"]),
            timestamp=float(row["timestamp"]), query=row["query"], answer=row["answer"],
            abstained=bool(row["abstained"]), abstain_reason=row["abstain_reason"],
            evidence_mass=float(row["evidence_mass"]), llm_calls=int(row["llm_calls"]),
            latency_ms=float(row["latency_ms"]), model=row["model"], flagged=bool(row["flagged"]),
            flag_reason=row["flag_reason"],
            flagged_at=float(row["flagged_at"]) if row["flagged_at"] is not None else None,
            cited_doc_ids=tuple(cited),
        )
