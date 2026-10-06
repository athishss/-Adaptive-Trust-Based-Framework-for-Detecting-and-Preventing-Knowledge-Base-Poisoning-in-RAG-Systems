#!/usr/bin/env python3
"""Download a BEIR dataset and select evaluation queries.

Downloads straight from Hugging Face (no account, no `datasets` library), with
resume support. By default the full corpus stays on disk as ``corpus.parquet``;
a deterministic sample of query records is written to ``queries_subset.jsonl``
without materializing corpus rows in memory. ``--subset`` is an optional,
explicitly smaller development mode.

    python scripts/download_data.py --dataset nq --out data/nq --queries 500
    python scripts/download_data.py --dataset nq --out data/nq --subset 200000 --queries 500

The first command retains all 2.68M NQ passages. The second creates a
gold-preserving corpus subset for quicker iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

HF = "https://huggingface.co/datasets/BeIR"

DATASETS: Dict[str, Dict[str, str]] = {
    "nq": {"corpus": f"{HF}/nq/resolve/main/corpus/corpus-00000-of-00001.parquet",
           "queries": f"{HF}/nq/resolve/main/queries/queries-00000-of-00001.parquet",
           "qrels": f"{HF}/nq-qrels/resolve/main/test.tsv",
           "passages": "2,681,468"},
    "hotpotqa": {"corpus": f"{HF}/hotpotqa/resolve/main/corpus/corpus-00000-of-00001.parquet",
                 "queries": f"{HF}/hotpotqa/resolve/main/queries/queries-00000-of-00001.parquet",
                 "qrels": f"{HF}/hotpotqa-qrels/resolve/main/test.tsv",
                 "passages": "5,233,329"},
    "msmarco": {"corpus": f"{HF}/msmarco/resolve/main/corpus/corpus-00000-of-00001.parquet",
                "queries": f"{HF}/msmarco/resolve/main/queries/queries-00000-of-00001.parquet",
                "qrels": f"{HF}/msmarco-qrels/resolve/main/dev.tsv",     # BEIR evaluates on dev
                "passages": "8,841,823"},
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(url: str, destination: Path, resume: bool = True) -> Path:
    """Stream a file to disk, resuming a partial download when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": "trace-rag/0.1"}
    if existing and resume:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length", 0)) + (
                existing if response.status == 206 else 0)
            if response.status == 206:
                print(f"  resuming {destination.name} at {human(existing)}")
                mode = "ab"
            else:
                mode = "wb"
                existing = 0
            done = existing
            with open(destination, mode) as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(f"\r  {destination.name}: {human(done)} / {human(total)} ({pct:5.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r  {destination.name}: {human(done)}", end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:          # already complete
            print(f"  {destination.name}: already complete ({human(existing)})")
            return destination
        raise SystemExit(f"download failed for {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc
    return destination


def read_qrels(path: Path) -> Dict[str, Set[str]]:
    """query-id -> {gold corpus ids} from a BEIR qrels TSV."""
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
            query_id, corpus_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                relevant = float(score) > 0
            except ValueError:
                continue
            if relevant:
                gold.setdefault(query_id, set()).add(corpus_id)
    return gold


def sample_query_ids(gold: Dict[str, Set[str]], n_queries: int,
                     seed: int = 20260921) -> List[str]:
    """Select a deterministic query sample from qrels without touching corpus rows."""
    if n_queries <= 0:
        raise ValueError("n_queries must be positive")
    query_ids = sorted(gold)
    random.Random(seed).shuffle(query_ids)
    return query_ids[:n_queries]


def build_subset(corpus_path: Path, gold_ids: Set[str], target_size: int, out_path: Path,
                 seed: int = 20260921) -> Tuple[int, int]:
    """Write a subset containing every gold passage plus a random fill."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("building a subset needs pyarrow: pip install pyarrow") from exc

    rng = random.Random(seed)
    parquet_file = pq.ParquetFile(str(corpus_path))
    total_rows = parquet_file.metadata.num_rows
    fill_target = max(0, target_size - len(gold_ids))
    keep_probability = min(1.0, fill_target / max(1, total_rows - len(gold_ids)))

    ids: List[str] = []
    titles: List[str] = []
    texts: List[str] = []
    gold_kept = 0
    for batch in parquet_file.iter_batches(batch_size=50_000, columns=["_id", "title", "text"]):
        rows = batch.to_pydict()
        for doc_id, title, text in zip(rows["_id"], rows["title"], rows["text"]):
            is_gold = doc_id in gold_ids
            if not is_gold and rng.random() > keep_probability:
                continue
            ids.append(doc_id)
            titles.append(title or "")
            texts.append(text or "")
            gold_kept += int(is_gold)
        print(f"\r  subset: {len(ids):,} passages kept ({gold_kept:,} gold)", end="", flush=True)
    print()
    table = pa.table({"_id": pa.array(ids), "title": pa.array(titles), "text": pa.array(texts)})
    pq.write_table(table, out_path)
    return len(ids), gold_kept


def write_queries_subset(queries_path: Path, query_ids: Sequence[str], out_path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("needs pyarrow: pip install pyarrow") from exc

    wanted = set(query_ids)
    rows = []
    parquet_file = pq.ParquetFile(str(queries_path))
    for batch in parquet_file.iter_batches(batch_size=50_000):
        data = batch.to_pylist()
        rows.extend(r for r in data if str(r.get("_id")) in wanted)
    out_path.write_text("\n".join(
        json.dumps({"qid": str(r["_id"]), "text": r.get("text", "")}) for r in rows), encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nq")
    parser.add_argument("--out", default=None, help="output directory (default: data/<dataset>)")
    parser.add_argument("--subset", type=int, default=None,
                        help="build a subset of this many passages (gold passages always kept)")
    parser.add_argument("--queries", type=int, default=500,
                        help="how many qrels-backed test queries to write for evaluation")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--skip-download", action="store_true", help="use files already on disk")
    args = parser.parse_args()
    if args.queries <= 0:
        parser.error("--queries must be positive")
    if args.subset is not None and args.subset <= 0:
        parser.error("--subset must be positive when supplied")

    spec = DATASETS[args.dataset]
    out_dir = Path(args.out or f"data/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.parquet"
    queries_path = out_dir / "queries.parquet"
    qrels_path = out_dir / "qrels.tsv"

    print(f"dataset: {args.dataset} ({spec['passages']} passages)\ndirectory: {out_dir.resolve()}")
    if not args.skip_download:
        print("downloading (resumable, safe to re-run):")
        download(spec["qrels"], qrels_path)
        download(spec["queries"], queries_path)
        download(spec["corpus"], corpus_path)

    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run without --skip-download")

    gold = read_qrels(qrels_path)
    print(f"qrels: {len(gold):,} test queries with gold passages")

    chosen_queries = sample_query_ids(gold, args.queries, seed=args.seed)
    corpus_for_run = corpus_path
    if args.subset:
        gold_ids: Set[str] = set()
        for query_id in chosen_queries:
            gold_ids |= gold[query_id]
        print(f"building a {args.subset:,}-passage subset covering {len(chosen_queries)} queries "
              f"({len(gold_ids):,} gold passages)")
        corpus_for_run = out_dir / "corpus_subset.parquet"
        kept, gold_kept = build_subset(corpus_path, gold_ids, args.subset, corpus_for_run, args.seed)
        missing = len(gold_ids) - gold_kept
        if missing:
            print(f"  WARNING: {missing} gold passages were not found in the corpus file")
        print(f"  wrote {kept:,} passages, including {gold_kept:,}/{len(gold_ids):,} gold passages")
    else:
        # Keep the complete corpus on disk. Sampling query IDs only reduces
        # evaluation traffic; it never materializes millions of passages.
        print(f"keeping all {spec['passages']} corpus passages; selecting "
              f"{len(chosen_queries)} evaluation queries")

    written = write_queries_subset(queries_path, chosen_queries, out_dir / "queries_subset.jsonl")
    print(f"wrote {written:,} queries to {out_dir / 'queries_subset.jsonl'}")

    # prove the result is readable by the ingestor before claiming success
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from trace_rag.ingestion import iter_beir_corpus

    sample = []
    for i, row in enumerate(iter_beir_corpus(corpus_for_run)):
        sample.append(row)
        if i >= 2:
            break
    print(f"\nreadable by the ingestor: {len(sample)} sample rows, first id = {sample[0]['doc_id']!r}")
    print(f"""
selected corpus: {corpus_for_run}
selected query file: {out_dir / 'queries_subset.jsonl'}

For full BEIR NQ, keep corpus.parquet (all 2,681,468 passages) and use the
query file only to bound evaluation traffic. config/nq_tpu.yaml uses Contriever
on TPU/XLA and a CPU FAISS IVF-PQ index. The query sample is deterministic;
this command does not claim a full-query benchmark.

For an explicitly smaller, gold-preserving development corpus, rerun with
--subset <passage-count>. That mode is not the full-data experiment.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
