"""Deterministic, exact-size BEIR subset construction and validation."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

_QRELS_HEADERS = {"query-id", "query_id", "qid", "query"}
_CORPUS_COLUMNS = ("_id", "title", "text")


def _identifier(value: Any, label: str) -> str:
    """Return a stripped, non-empty string identifier or fail loudly."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty identifier")
    return text


def _atomic_write(path: Path, write) -> None:
    """Run ``write(tmp_path)`` and atomically install the result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def select_query_ids(gold: Mapping[str, Set[str]], count: int,
                     seed: int = 20260921) -> List[str]:
    """Choose exactly ``count`` query ids deterministically, without repeats."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("query count must be an integer")
    if count < 1:
        raise ValueError("query count must be positive")
    candidates = []
    for raw in gold:
        qid = _identifier(raw, "qrels query id")
        if qid in candidates:
            raise ValueError(f"qrels contains the duplicate query id {qid!r}")
        candidates.append(qid)
    if len(candidates) < count:
        raise ValueError(f"requested {count} queries but qrels has only {len(candidates)}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def read_qrels_set(path: Path) -> Dict[str, Set[str]]:
    """query-id -> positive-relevance corpus-id set (header optional).

    Blank lines are ignored.  Malformed rows and non-finite or non-numeric
    scores are rejected with the line number instead of being silently
    dropped, because a silently short qrels file would weaken evaluation.
    """
    gold: Dict[str, Set[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter="\t")
        first_row = True
        for line_no, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if first_row:
                first_row = False
                if row[0].strip().lower() in _QRELS_HEADERS:
                    continue
            if len(row) < 3:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 3 tab-separated qrels fields")
            qid = _identifier(row[0], f"{path}:{line_no}: query id")
            doc_id = _identifier(row[1], f"{path}:{line_no}: corpus id")
            try:
                score = float(row[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: score is not numeric: {row[2]!r}") from exc
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_no}: score must be finite, got {row[2]!r}")
            if score > 0:
                gold.setdefault(qid, set()).add(doc_id)
    return gold


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("subset construction needs pyarrow: pip install pyarrow") from exc
    return pa, pq


def build_exact_subset(corpus_path: Path, gold_ids: Set[str], target_size: int,
                       out_path: Path, seed: int = 20260921) -> Tuple[int, int]:
    """Write exactly ``target_size`` unique rows, retaining all ``gold_ids``.

    The corpus is streamed with bounded memory: only the required gold rows
    and a reservoir of ``target_size - len(gold_ids)`` non-gold rows are held.
    The output is written atomically and every selected id is checked for
    uniqueness before installation.
    """
    if isinstance(target_size, bool) or not isinstance(target_size, int):
        raise TypeError("target size must be an integer")
    if target_size < 1:
        raise ValueError("target size must be positive")
    wanted = {_identifier(doc_id, "gold doc id") for doc_id in gold_ids}
    if target_size < len(wanted):
        raise ValueError(
            f"target size {target_size} is smaller than the {len(wanted)} required gold passages; "
            "refusing to truncate gold evidence")
    pa, pq = _pyarrow()
    parquet_file = pq.ParquetFile(str(corpus_path))
    missing_columns = [name for name in _CORPUS_COLUMNS
                       if name not in parquet_file.schema_arrow.names]
    if missing_columns:
        raise ValueError(f"corpus parquet is missing required columns: {missing_columns}")
    fill_target = target_size - len(wanted)
    rng = random.Random(seed)
    gold_rows: Dict[str, Tuple[int, str, str, str]] = {}
    reservoir: List[Tuple[int, str, str, str]] = []
    non_gold_seen = 0
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=50_000, columns=list(_CORPUS_COLUMNS)):
        rows = batch.to_pydict()
        for raw_id, title, text in zip(rows["_id"], rows["title"], rows["text"]):
            doc_id = _identifier(raw_id, f"corpus row {row_index} document id")
            record = (row_index, doc_id, title or "", text or "")
            row_index += 1
            if doc_id in wanted:
                if doc_id in gold_rows:
                    raise ValueError(
                        f"corpus contains the duplicate gold document id {doc_id!r}; "
                        "refusing an ambiguous subset")
                gold_rows[doc_id] = record
                continue
            non_gold_seen += 1
            if len(reservoir) < fill_target:
                reservoir.append(record)
            elif fill_target:
                replacement = rng.randrange(non_gold_seen)
                if replacement < fill_target:
                    reservoir[replacement] = record
    missing = wanted - set(gold_rows)
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} gold passages are absent from the corpus: {sample}")
    if fill_target > non_gold_seen:
        raise ValueError(
            f"corpus has only {non_gold_seen + len(gold_rows):,} rows; cannot build {target_size:,}")
    selected = list(gold_rows.values()) + reservoir
    if len(selected) != target_size:
        raise RuntimeError(f"subset construction produced {len(selected)} rows, expected {target_size}")
    selected_ids = [row[1] for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("corpus contains duplicate selected document ids; "
                         "refusing an ambiguous subset")
    selected.sort(key=lambda item: item[0])
    table = pa.table({"_id": pa.array([row[1] for row in selected]),
                      "title": pa.array([row[2] for row in selected]),
                      "text": pa.array([row[3] for row in selected])})
    _atomic_write(out_path, lambda temporary: pq.write_table(table, temporary))
    return len(selected), len(gold_rows)


def write_queries_exact(queries_path: Path, query_ids: Sequence[str], out_path: Path) -> int:
    """Write exactly the requested unique query ids and fail if any is missing."""
    _, pq = _pyarrow()
    wanted: List[str] = []
    seen: Set[str] = set()
    for raw in query_ids:
        qid = _identifier(raw, "selected query id")
        if qid in seen:
            raise ValueError(f"selected query ids contain the duplicate id {qid!r}")
        seen.add(qid)
        wanted.append(qid)
    if not wanted:
        raise ValueError("selected query ids must not be empty")
    parquet_file = pq.ParquetFile(str(queries_path))
    missing_columns = [name for name in ("_id", "text")
                       if name not in parquet_file.schema_arrow.names]
    if missing_columns:
        raise ValueError(f"queries parquet is missing required columns: {missing_columns}")
    found: Dict[str, Dict[str, Any]] = {}
    for batch in parquet_file.iter_batches(batch_size=50_000, columns=["_id", "text"]):
        rows = batch.to_pydict()
        for raw_id, text in zip(rows["_id"], rows["text"]):
            qid = "" if raw_id is None else str(raw_id).strip()
            if qid not in seen:
                continue
            if qid in found:
                raise ValueError(f"queries parquet contains the duplicate query id {qid!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"query {qid!r} has no non-empty text")
            found[qid] = {"qid": qid, "text": text}
    missing = seen - set(found)
    if missing:
        raise ValueError(f"{len(missing)} selected queries are absent from the queries file: "
                         f"{', '.join(sorted(missing)[:10])}")
    payload = "\n".join(json.dumps(found[qid], ensure_ascii=False) for qid in wanted) + "\n"
    _atomic_write(out_path, lambda temporary: temporary.write_text(payload, encoding="utf-8"))
    return len(wanted)


def verify_subset(corpus_path: Path, queries_path: Path, qrels_path: Path,
                  expected_passages: int, expected_queries: int) -> Dict[str, Any]:
    """Validate actual row counts, query membership and 100% gold coverage."""
    for label, path in (("corpus", corpus_path), ("queries", queries_path), ("qrels", qrels_path)):
        if not path.exists():
            return {"ok": False, "errors": [f"missing {label}: {path}"]}
    _, pq = _pyarrow()
    errors: List[str] = []
    try:
        corpus_parquet = pq.ParquetFile(str(corpus_path))
        corpus_rows = int(corpus_parquet.metadata.num_rows)
    except Exception as exc:
        return {"ok": False, "errors": [f"cannot read corpus parquet: {type(exc).__name__}: {exc}"]}
    query_ids: List[str] = []
    seen_queries: Set[str] = set()
    try:
        query_lines = queries_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"ok": False, "errors": [f"cannot read query subset: {type(exc).__name__}: {exc}"]}
    for line_no, line in enumerate(query_lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"query line {line_no} is not valid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"query line {line_no} is not a JSON object")
            continue
        raw_qid = row.get("qid") if row.get("qid") is not None else row.get("_id")
        qid = "" if raw_qid is None else str(raw_qid).strip()
        if not qid:
            errors.append(f"query line {line_no} has no non-empty qid")
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"query {qid!r} has no non-empty text")
        if qid in seen_queries:
            errors.append(f"query subset contains the duplicate qid {qid!r}")
            continue
        seen_queries.add(qid)
        query_ids.append(qid)
    if len(query_ids) != expected_queries:
        errors.append(f"query subset has {len(query_ids)} rows, expected {expected_queries}")
    try:
        gold = read_qrels_set(qrels_path)
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": errors + [
            f"cannot read qrels: {type(exc).__name__}: {exc}"]}
    selected_gold: Set[str] = set()
    missing_query_gold: List[str] = []
    for qid in query_ids:
        ids = gold.get(qid, set())
        if not ids:
            missing_query_gold.append(qid)
        selected_gold |= ids
    if missing_query_gold:
        errors.append(f"{len(missing_query_gold)} selected queries have no positive qrels")
    corpus_ids: Set[str] = set()
    duplicate_ids = 0
    try:
        for batch in corpus_parquet.iter_batches(batch_size=100_000, columns=["_id"]):
            for raw_id in batch.column(0).to_pylist():
                doc_id = "" if raw_id is None else str(raw_id).strip()
                if not doc_id:
                    errors.append("corpus subset contains an empty document id")
                    continue
                if doc_id in corpus_ids:
                    duplicate_ids += 1
                corpus_ids.add(doc_id)
    except Exception as exc:
        return {"ok": False, "errors": errors + [
            f"cannot read corpus ids: {type(exc).__name__}: {exc}"]}
    if duplicate_ids:
        errors.append(f"corpus subset contains {duplicate_ids} duplicate ids")
    if corpus_rows != expected_passages:
        errors.append(f"corpus has {corpus_rows:,} rows, expected exactly {expected_passages:,}")
    missing_gold = selected_gold - corpus_ids
    if missing_gold:
        errors.append(f"{len(missing_gold)} required gold passages are absent from the corpus")
    coverage = ((len(selected_gold) - len(missing_gold)) / len(selected_gold)
                if selected_gold else 0.0)
    return {
        "ok": not errors,
        "errors": errors,
        "corpus_rows": corpus_rows,
        "expected_corpus_rows": expected_passages,
        "query_rows": len(query_ids),
        "expected_queries": expected_queries,
        "gold_passages_required": len(selected_gold),
        "gold_passages_present": len(selected_gold) - len(missing_gold),
        "gold_coverage": coverage,
        "missing_gold": sorted(missing_gold)[:20],
        "missing_query_gold": missing_query_gold[:20],
        "duplicate_corpus_ids": duplicate_ids,
    }
