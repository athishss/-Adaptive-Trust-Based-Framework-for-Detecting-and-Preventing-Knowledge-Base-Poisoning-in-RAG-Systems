"""Load BEIR-style corpus/query/qrels files into the shared experiment fixture.

An optional answer file supplies evaluator-only gold answers keyed by query id.
When it is supplied, every selected target must have a non-empty answer; a
partial answer file is rejected rather than silently producing a mixed
MEASURED/NOT MEASURED ASR denominator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _answers_from_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = [str(value)]
    out: List[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_answer_map(path: str | Path) -> Dict[str, List[str]]:
    """Read qid -> list-of-gold-answers from JSON or JSONL.

    Accepted row keys are ``gold_answers``, ``answers``, ``answer`` and
    ``gold_answer``. A JSON object mapping qid directly to a string/list is
    also accepted.
    """
    answer_path = Path(path)
    if not answer_path.exists():
        raise FileNotFoundError(f"answer file does not exist: {answer_path}")
    text = answer_path.read_text(encoding="utf-8").strip()
    mapping: Dict[str, List[str]] = {}
    if not text:
        raise ValueError(f"answer file is empty: {answer_path}")
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("answer JSON must be an object or JSONL rows")
        # A single JSONL-style row has qid plus answer fields; a mapping has
        # qid keys that are not reserved answer-field names.
        qid = payload.get("qid") or payload.get("_id") or payload.get("query_id")
        has_answer_field = any(key in payload for key in
                               ("gold_answers", "answers", "answer", "gold_answer"))
        if qid and has_answer_field:
            answers = _answers_from_value(
                payload.get("gold_answers") or payload.get("answers")
                or payload.get("answer") or payload.get("gold_answer"))
            if answers:
                mapping[str(qid)] = answers
        else:
            # A mapping of qid -> answer(s), optionally wrapped in {"answers": ...}.
            items = payload.get("answers") if isinstance(payload.get("answers"), dict) else payload
            for qid, value in items.items():
                answers = _answers_from_value(value)
                if answers:
                    mapping[str(qid)] = answers
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qid") or row.get("_id") or row.get("query_id")
            answers = _answers_from_value(
                row.get("gold_answers") or row.get("answers")
                or row.get("answer") or row.get("gold_answer"))
            if not qid:
                raise ValueError("every answer row needs qid/_id/query_id")
            if not answers:
                raise ValueError(f"answer row {qid!r} has no non-empty answer")
            mapping[str(qid)] = answers
    if not mapping:
        raise ValueError(f"answer file contains no usable answers: {answer_path}")
    return mapping


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
                      target_limit: int = 20, seed: int = 20260921,
                      answers: Optional[str | Path] = None) -> Fixture:
    """Build a deterministic BEIR fixture without reading hidden attack labels.

    ``answers`` is evaluator-only. If provided, coverage is strict: missing or
    empty answers for any selected target raise before a run starts.
    """
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
            "gold_answers": [],
            "gold_doc_id": candidates[0],
        })
        if len(selected) >= int(target_limit):
            break
    if not selected:
        raise ValueError("no query with a gold passage in the corpus subset")
    answer_map: Dict[str, List[str]] = {}
    if answers is not None:
        answer_map = load_answer_map(answers)
        missing = [row["qid"] for row in selected if not answer_map.get(row["qid"])]
        if missing:
            raise ValueError(
                "answer file is missing gold answers for selected targets: "
                + ", ".join(missing[:20])
                + f" ({len(missing)} missing)")
        for row in selected:
            row["gold_answers"] = list(answer_map[row["qid"]])
            row["gold_answer"] = row["gold_answers"][0]
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


__all__ = ["load_beir_fixture", "load_answer_map"]
