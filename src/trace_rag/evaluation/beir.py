"""Load BEIR-style corpus/query/qrels files into the shared experiment fixture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .experiment import Fixture


def _rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return list(payload.get("questions") or payload.get("queries") or [])
        return list(payload)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("reading a Parquet corpus requires pyarrow") from exc
    table = pq.read_table(path)
    return table.to_pylist()


def _qrels(path: Path) -> Dict[str, List[str]]:
    gold: Dict[str, set] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0].lower() in {"query-id", "qid"}:
            continue
        try:
            score = float(parts[2])
        except ValueError:
            continue
        if score > 0:
            gold.setdefault(str(parts[0]), set()).add(str(parts[1]))
    return {qid: sorted(ids) for qid, ids in gold.items()}


def load_beir_fixture(corpus: str | Path, queries: str | Path, qrels: str | Path,
                      target_limit: int = 20, seed: int = 20260921) -> Fixture:
    """Build a deterministic BEIR fixture without reading hidden attack labels."""
    corpus_path, query_path, qrel_path = Path(corpus), Path(queries), Path(qrels)
    rows = _rows(corpus_path)
    by_id = {str(row.get("_id") or row.get("id")): row for row in rows}
    qrows = _rows(query_path)
    gold = _qrels(qrel_path)
    selected = []
    for row in qrows:
        qid = str(row.get("_id") or row.get("id") or row.get("qid"))
        candidates = [doc_id for doc_id in gold.get(qid, []) if doc_id in by_id]
        if not candidates:
            continue
        selected.append({
            "qid": qid,
            "question": str(row.get("text") or row.get("question") or ""),
            "gold_answer": "",
            "gold_doc_id": candidates[0],
        })
        if len(selected) >= int(target_limit):
            break
    if not selected:
        raise ValueError("no query with a gold passage in the corpus subset")
    clean: List[Dict[str, Any]] = []
    for row in rows:
        doc_id = str(row.get("_id") or row.get("id"))
        digest = hashlib.sha256(doc_id.encode()).hexdigest()
        clean.append({
            "_id": doc_id,
            "text": str(row.get("text") or ""),
            "title": str(row.get("title") or ""),
            "source_id": f"beir_source_{int(digest[:8], 16) % 20:02d}",
        })
    false_answers = {row["qid"]: f"Fabricated answer {i + 1}" for i, row in enumerate(selected)}
    return Fixture(clean=clean, late_clean=[], questions=selected,
                   targets=[row["qid"] for row in selected], false_answers=false_answers)

__all__ = ["load_beir_fixture"]
