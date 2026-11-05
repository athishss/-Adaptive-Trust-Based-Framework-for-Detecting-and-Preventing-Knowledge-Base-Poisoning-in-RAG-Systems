#!/usr/bin/env python3
"""Download a BEIR dataset and prepare a working subset.

Downloads straight from Hugging Face (no account, no `datasets` library), with
resume support, then optionally builds a development subset.

Why the subset is not a random sample: a random slice of 50k passages from
2.68M would drop almost every *gold* passage, so retrieval would look broken
for reasons that have nothing to do with the defence.  ``--subset`` keeps every
gold passage for the sampled queries and fills the rest at random, which is the
standard way to shrink a retrieval benchmark honestly.

    python scripts/download_data.py --dataset nq --out data/nq
    python scripts/download_data.py --dataset nq --out data/nq --subset 200000 --queries 500

Then:

    trace-rag --config config/full_nq.yaml ingest-beir --corpus data/nq/corpus_subset.parquet
    trace-rag --config config/full_nq.yaml index
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

ROOT = Path(__file__).resolve().parent.parent


def _subset_helpers():
    root = str(ROOT / "src")
    if root not in sys.path:
        sys.path.insert(0, root)
    from trace_rag.ingestion.subset import (build_exact_subset, select_query_ids,
                                            verify_subset, write_queries_exact)
    return build_exact_subset, select_query_ids, verify_subset, write_queries_exact

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


def build_subset(corpus_path: Path, gold_ids: Set[str], target_size: int, out_path: Path,
                 seed: int = 20260921) -> Tuple[int, int]:
    """Write exactly ``target_size`` rows, retaining every gold passage."""
    build_exact_subset, _, _, _ = _subset_helpers()
    return build_exact_subset(corpus_path, set(gold_ids), target_size, out_path, seed)


def write_queries_subset(queries_path: Path, query_ids: Sequence[str], out_path: Path) -> int:
    _, _, _, write_queries_exact = _subset_helpers()
    return write_queries_exact(queries_path, query_ids, out_path)


def verify_subset(corpus_path: Path, queries_path: Path, qrels_path: Path,
                  expected_passages: int, expected_queries: int) -> Dict:
    _, _, verify, _ = _subset_helpers()
    return verify(corpus_path, queries_path, qrels_path, expected_passages, expected_queries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nq")
    parser.add_argument("--out", default=None, help="output directory (default: data/<dataset>)")
    parser.add_argument("--subset", type=int, default=None,
                        help="build a subset of this many passages (gold passages always kept)")
    parser.add_argument("--queries", type=int, default=500,
                        help="how many test queries the subset must cover")
    parser.add_argument("--include-queries", default=None, metavar="JSON",
                        help="always keep these queries (and their gold passages) in the subset: a JSON "
                             "dict/list whose keys or 'id' fields are BEIR query ids, e.g. PoisonedRAG's "
                             "results/adv_targeted_results/nq.json. The query count is raised to cover them.")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--skip-download", action="store_true", help="use files already on disk")
    args = parser.parse_args()

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

    corpus_for_run = corpus_path
    if args.subset:
        _, select_query_ids, _, _ = _subset_helpers()
        forced: List[str] = []
        if args.include_queries:
            payload = json.loads(Path(args.include_queries).read_text(encoding="utf-8"))
            rows = payload.values() if isinstance(payload, dict) else payload
            wanted = [str(r["id"]) if isinstance(r, dict) else str(r) for r in rows]
            forced = [q for q in dict.fromkeys(wanted) if q in gold]
            print(f"forcing {len(forced)}/{len(set(wanted))} queries from {args.include_queries} "
                  "into the subset (the rest have no qrels entry and cannot be attacked)")
        n_queries = max(args.queries, len(forced))
        chosen_queries = select_query_ids(gold, n_queries, args.seed)
        if forced:
            chosen_queries = forced + [q for q in chosen_queries if q not in set(forced)]
            chosen_queries = chosen_queries[:n_queries]
        gold_ids: Set[str] = set()
        for query_id in chosen_queries:
            gold_ids |= gold[query_id]
        print(f"building exactly {args.subset:,} passages covering {len(chosen_queries)} queries "
              f"({len(gold_ids):,} required gold passages)")
        corpus_for_run = out_dir / "corpus_subset.parquet"
        kept, gold_kept = build_subset(corpus_path, gold_ids, args.subset, corpus_for_run, args.seed)
        written = write_queries_subset(queries_path, chosen_queries, out_dir / "queries_subset.jsonl")
        contract = verify_subset(corpus_for_run, out_dir / "queries_subset.jsonl", qrels_path,
                                 args.subset, n_queries)
        if not contract["ok"]:
            raise SystemExit("subset verification failed: " + "; ".join(contract["errors"]))
        print(f"  wrote {kept:,} passages and {written} queries; gold coverage "
              f"{contract['gold_passages_present']}/{contract['gold_passages_required']} "
              f"({contract['gold_coverage']:.0%})")

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
next (config/nq_gpu.yaml uses Contriever on the GPU and an exact FAISS index):

  trace-rag --config config/nq_gpu.yaml ingest-beir --corpus {corpus_for_run}
  trace-rag --config config/nq_gpu.yaml index
  trace-rag --config config/nq_gpu.yaml query "who designed the eiffel tower?" --set generation.backend=stub

no GPU?   add            --set embedding.device=cpu      (slower; try a smaller --subset first)
no LLM?   keep           --set generation.backend=stub   until vLLM or Ollama is running
full corpus (2.68M)?     use config/full_nq.yaml, which switches the index to IVF-PQ
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
