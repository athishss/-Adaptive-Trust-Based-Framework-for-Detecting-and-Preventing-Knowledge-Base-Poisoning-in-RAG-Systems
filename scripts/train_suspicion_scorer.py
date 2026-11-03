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
from trace_rag.evaluation.experiment import build_case_plan, build_pipeline, load_fixture  # noqa: E402
from trace_rag.ingestion.pipeline import Ingestor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=str(ROOT / "examples"))
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attacks", default="entity_swap,negation,instruction_injection,sybil,adaptive")
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    fixture = load_fixture(args.examples)
    rows = []
    for attack in [x for x in args.attacks.split(",") if x]:
        plan = build_case_plan(fixture, attack, seed=args.seed,
                               target_limit=args.targets, repeat_targets=2)
        root = Path(tempfile.mkdtemp(prefix=f"train_{attack}_"))
        pipeline = build_pipeline(args.config, root, "no_defence", fixture)
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
