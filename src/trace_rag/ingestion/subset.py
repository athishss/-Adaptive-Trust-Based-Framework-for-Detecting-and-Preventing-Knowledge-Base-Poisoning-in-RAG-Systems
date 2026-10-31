"""Deterministic, exact-size BEIR subset construction and validation."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple


def select_query_ids(gold: Mapping[str, Set[str]], count: int,
                     seed: int = 20260921) -> List[str]:
    """Choose exactly ``count`` query ids deterministically, without repeats."""
    if count < 1:
        raise ValueError("query count must be positive")
    candidates = sorted(gold)
    if len(candidates) < count:
        raise ValueError(f"requested {count} queries but qrels has only {len(candidates)}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def read_qrels_set(path: Path) -> Dict[str, Set[str]]:
    """query-id -> positive-relevance corpus-id set (header optional)."""
    gold: Dict[str, Set[str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header and header[0].strip() not in {"query-id", "query_id"}:
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                relevant = float(score) > 0
            except ValueError:
                continue
            if relevant:
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
    """Write exactly ``target_size`` rows, retaining all ``gold_ids``."""
    if target_size < len(gold_ids):
        raise ValueError(
            f"target size {target_size} is smaller than the {len(gold_ids)} required gold passages; "
            "refusing to truncate gold evidence")
    pa, pq = _pyarrow()
    parquet_file = pq.ParquetFile(str(corpus_path))
    fill_target = target_size - len(gold_ids)
    rng = random.Random(seed)
    gold_rows: Dict[str, Tuple[int, str, str, str]] = {}
    reservoir: List[Tuple[int, str, str, str]] = []
    non_gold_seen = 0
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=50_000, columns=["_id", "title", "text"]):
        rows = batch.to_pydict()
        for doc_id, title, text in zip(rows["_id"], rows["title"], rows["text"]):
            doc_id = str(doc_id)
            record = (row_index, doc_id, title or "", text or "")
            row_index += 1
            if doc_id in gold_ids:
                gold_rows[doc_id] = record
                continue
            non_gold_seen += 1
            if len(reservoir) < fill_target:
                reservoir.append(record)
            elif fill_target:
                replacement = rng.randrange(non_gold_seen)
                if replacement < fill_target:
                    reservoir[replacement] = record
    missing = gold_ids - set(gold_rows)
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} gold passages are absent from the corpus: {sample}")
    if fill_target > non_gold_seen:
        raise ValueError(
            f"corpus has only {non_gold_seen + len(gold_rows):,} rows; cannot build {target_size:,}")
    selected = list(gold_rows.values()) + reservoir
    if len(selected) != target_size:
        raise RuntimeError(f"subset construction produced {len(selected)} rows, expected {target_size}")
    selected.sort(key=lambda item: item[0])
    table = pa.table({"_id": pa.array([row[1] for row in selected]),
                      "title": pa.array([row[2] for row in selected]),
                      "text": pa.array([row[3] for row in selected])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return len(selected), len(gold_rows)


def write_queries_exact(queries_path: Path, query_ids: Sequence[str], out_path: Path) -> int:
    """Write exactly the requested query ids and fail if any is missing."""
    _, pq = _pyarrow()
    wanted = list(dict.fromkeys(str(q) for q in query_ids))
    wanted_set = set(wanted)
    found: Dict[str, Dict[str, Any]] = {}
    for batch in pq.ParquetFile(str(queries_path)).iter_batches(batch_size=50_000):
        for row in batch.to_pylist():
            qid = str(row.get("_id"))
            if qid in wanted_set:
                found[qid] = {"qid": qid, "text": row.get("text", "")}
    missing = wanted_set - set(found)
    if missing:
        raise ValueError(f"{len(missing)} selected queries are absent from the queries file: "
                         f"{', '.join(sorted(missing)[:10])}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(found[qid], ensure_ascii=False) for qid in wanted) + "\n",
                        encoding="utf-8")
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
        corpus_rows = int(pq.ParquetFile(str(corpus_path)).metadata.num_rows)
    except Exception as exc:
        return {"ok": False, "errors": [f"cannot read corpus parquet: {type(exc).__name__}: {exc}"]}
    query_ids: List[str] = []
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            query_ids.append(str(row.get("qid") or row.get("_id")))
    if len(query_ids) != len(set(query_ids)):
        errors.append("query subset contains duplicate qids")
    if len(query_ids) != expected_queries:
        errors.append(f"query subset has {len(query_ids)} rows, expected {expected_queries}")
    gold = read_qrels_set(qrels_path)
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
    for batch in pq.ParquetFile(str(corpus_path)).iter_batches(batch_size=100_000, columns=["_id"]):
        for doc_id in batch.column(0).to_pylist():
            doc_id = str(doc_id)
            if doc_id in corpus_ids:
                duplicate_ids += 1
            corpus_ids.add(doc_id)
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
