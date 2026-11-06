#!/usr/bin/env python3
"""Build leakage-guarded training rows and fit the CPU logistic scorer."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from trace_rag.config import Config  # noqa: E402
from trace_rag.detection.training import TrainingSet, train_scorer  # noqa: E402
from trace_rag.evaluation.chronological import run_stream  # noqa: E402
from trace_rag.evaluation.experiment import (OFFICIAL_ATTACKS, build_case_plan,  # noqa: E402
                                             build_pipeline, load_fixture)
from trace_rag.evaluation.poisonedrag import load_poisonedrag_fixture  # noqa: E402
from trace_rag.ingestion.pipeline import Ingestor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=str(ROOT / "examples"))
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attacks", default="entity_swap,negation,instruction_injection,sybil,adaptive")
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260921)
    real = ap.add_argument_group(
        "real data (train on disjoint real NQ questions, test on the published poison)")
    real.add_argument("--poisonedrag-json", help="PoisonedRAG adv_targeted_results/<dataset>.json; its "
                      "questions give real gold passages. Training poison is still produced by the "
                      "template attacks, never by the published adv_texts.")
    real.add_argument("--corpus")
    real.add_argument("--qrels")
    real.add_argument("--max-clean", type=int, default=3000)
    real.add_argument("--target-offset", type=int, default=0,
                      help="first training question; set it past the evaluation targets so the "
                           "scorer never sees an evaluated question")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    banned = [a for a in args.attacks.split(",") if a in OFFICIAL_ATTACKS]
    if banned:
        raise SystemExit(f"refusing to train on {banned}: those are the evaluation attacks")
    if args.poisonedrag_json:
        if not (args.corpus and args.qrels):
            raise SystemExit("--poisonedrag-json needs --corpus and --qrels")
        fixture, fx_report = load_poisonedrag_fixture(
            args.poisonedrag_json, args.corpus, args.qrels, target_limit=args.targets,
            target_offset=args.target_offset, max_clean=args.max_clean, seed=args.seed)
        print(json.dumps({"training_fixture": fx_report}, indent=2))
    else:
        fixture = load_fixture(args.examples)
    rows = []
    for attack in [x for x in args.attacks.split(",") if x]:
        plan = build_case_plan(fixture, attack, seed=args.seed,
                               target_limit=args.targets, repeat_targets=2)
        root = Path(tempfile.mkdtemp(prefix=f"train_{attack}_"))
        pipeline = build_pipeline(args.config, root, "no_defence", fixture,
                                  overrides=args.overrides)
        try:
            run = run_stream(pipeline, Ingestor(pipeline.store, pipeline.config.ingestion,
                                                clock=pipeline.clock),
                             plan.ingestions, plan.queries, log=False)
            rows.extend(run.training_rows)
        finally:
            pipeline.close()
    dataset = TrainingSet()
    dataset.extend(rows)
    if len(dataset) < 12:
        raise RuntimeError("training rows are insufficient; increase --targets/--attacks")
    config = Config.load(args.config)
    scorer, report, held_out = train_scorer(dataset, config.scorer, seed=args.seed)
    out = Path(args.out)
    scorer.save(out)
    (out.with_suffix(".report.json")).write_text(json.dumps({
        "training": dataset.summary(), "thresholds": report.to_dict(),
        "held_out_test": held_out, "config": args.__dict__,
        "note": "CPU logistic regression; no test labels enter fit or calibration",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(dataset), "scorer": str(out),
                      "thresholds": report.to_dict(), "held_out": held_out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
