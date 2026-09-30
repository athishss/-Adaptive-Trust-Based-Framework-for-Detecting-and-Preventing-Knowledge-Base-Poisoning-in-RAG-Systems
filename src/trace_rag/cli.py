"""Command line interface (stdlib argparse, no extra dependency).

    trace-rag ingest      --root data/docs
    trace-rag ingest-beir --corpus data/nq/corpus.jsonl --limit 50000
    trace-rag index
    trace-rag query       "who designed the eiffel tower?"
    trace-rag train-scorer --rows runs/default/labelled_rows.jsonl
    trace-rag remediate   --doc-ids p1#0000 p2#0000
    trace-rag stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import Config
from .detection.scorer import SuspicionScorer
from .detection.training import LabelledRow, TrainingSet, train_scorer
from .contracts import FeatureSnapshot, SignalVector
from .ingestion.pipeline import FixedSourceAssigner, Ingestor, SourceAssigner
from .pipeline import PersonAPipeline
from .utils.logging import get_logger

logger = get_logger("trace_rag.cli")


def _pipeline(args: argparse.Namespace, load_index: bool = True) -> PersonAPipeline:
    config = Config.load(args.config) if args.config else Config()
    if args.root:
        config.storage.root = args.root
    return PersonAPipeline.from_config(config, load_existing_index=load_index)


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, load_index=False)
    ingestor = Ingestor(pipeline.store, pipeline.config.ingestion)
    report = ingestor.ingest_paths(args.path, recursive=not args.no_recursive)
    print(json.dumps(report.to_dict(), indent=2))
    pipeline.close()
    return 0


def cmd_ingest_beir(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, load_index=False)
    ingestor = Ingestor(pipeline.store, pipeline.config.ingestion)
    assigner = (FixedSourceAssigner(args.source_id) if args.source_id
                else SourceAssigner(n_sources=args.n_sources, seed=pipeline.config.seed))
    report = ingestor.ingest_beir(args.corpus, assigner, limit=args.limit,
                                  seconds_per_doc=args.seconds_per_doc)
    print(json.dumps(report.to_dict(), indent=2))
    pipeline.close()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, load_index=not args.rebuild)
    count = pipeline.index_chunks()
    path = pipeline.save_index()
    print(json.dumps({"indexed": count, "index_size": len(pipeline.index), "path": path}, indent=2))
    pipeline.close()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    result = pipeline.answer(args.question, query_id=args.query_id)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\nQ: {result.query}\nA: {result.answer}\n")
        if result.record.abstained:
            print(f"   abstained: {result.record.abstain_reason}")
        print(f"   evidence mass {result.record.evidence_mass:.2f}, "
              f"{result.llm_calls} LLM call(s), {result.timings_ms.get('total', 0):.1f} ms")
        print("   passages:")
        for assessment in result.assessments:
            marker = {"LOW": " ", "MEDIUM": "?", "HIGH": "!"}[assessment.band.value]
            print(f"    {marker} {assessment.doc_id:<28} suspicion={assessment.suspicion:.3f} "
                  f"{assessment.band.value:<6} {assessment.signals.to_dict()}")
    pipeline.close()
    return 0


def cmd_train_scorer(args: argparse.Namespace) -> int:
    """Train from a JSONL of labelled rows produced by Person C's harness."""
    config = Config.load(args.config) if args.config else Config()
    dataset = TrainingSet()
    with open(args.rows, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            features = row["features"]
            signals = SignalVector(**{k: float(v) for k, v in features.items()
                                      if k.startswith("s")})
            extras = {k: float(v) for k, v in features.items() if k.startswith("x_")}
            dataset.add(LabelledRow(
                query_id=str(row["query_id"]), doc_id=str(row["doc_id"]), label=int(row["label"]),
                features=FeatureSnapshot(signals=signals, extras=extras),
                attack_family=str(row.get("attack_family", "none")),
                captured_at=float(row.get("captured_at", 0.0)),
                query_time=float(row.get("query_time", 0.0)),
            ))
    scorer, report, held_out = train_scorer(dataset, config.scorer)
    out = args.out or str(config.path(config.storage.scorer_path))
    scorer.save(out)
    print(json.dumps({"dataset": dataset.summary(), "thresholds": report.to_dict(),
                      "held_out_test": held_out, "saved_to": out}, indent=2, default=float))
    return 0


def cmd_remediate(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    report = pipeline.on_quarantine(args.doc_ids, reason=args.reason)
    print(json.dumps(report.to_dict(), indent=2))
    pipeline.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    print(json.dumps(pipeline.stats(), indent=2))
    pipeline.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-rag", description="TRACE-RAG Person A pipeline")
    parser.add_argument("--config", help="path to a YAML config file")
    parser.add_argument("--root", help="override storage.root")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest PDF/TXT/MD/HTML files from a folder")
    p_ingest.add_argument("--path", required=True)
    p_ingest.add_argument("--no-recursive", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_beir = sub.add_parser("ingest-beir", help="ingest a BEIR corpus.jsonl")
    p_beir.add_argument("--corpus", required=True)
    p_beir.add_argument("--limit", type=int, default=None)
    p_beir.add_argument("--n-sources", type=int, default=2000)
    p_beir.add_argument("--source-id", default=None, help="force one source id (attacker corpora)")
    p_beir.add_argument("--seconds-per-doc", type=float, default=60.0)
    p_beir.set_defaults(func=cmd_ingest_beir)

    p_index = sub.add_parser("index", help="embed and index everything in the store")
    p_index.add_argument("--rebuild", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="answer one question")
    p_query.add_argument("question")
    p_query.add_argument("--query-id", default="cli")
    p_query.add_argument("--json", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_train = sub.add_parser("train-scorer", help="train the suspicion scorer from labelled rows")
    p_train.add_argument("--rows", required=True)
    p_train.add_argument("--out", default=None)
    p_train.set_defaults(func=cmd_train_scorer)

    p_rem = sub.add_parser("remediate", help="flag past answers that used quarantined passages")
    p_rem.add_argument("--doc-ids", nargs="+", required=True)
    p_rem.add_argument("--reason", default="quarantined by trust ledger")
    p_rem.set_defaults(func=cmd_remediate)

    p_stats = sub.add_parser("stats", help="show pipeline statistics")
    p_stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:                                 # pragma: no cover
        return 130
    except Exception as exc:                                  # pragma: no cover - CLI guard
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
