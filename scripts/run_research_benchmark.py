#!/usr/bin/env python3
"""Run a real dataset through the chronological attack/baseline matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from trace_rag.evaluation.beir import load_beir_fixture  # noqa: E402
from trace_rag.evaluation.experiment import SMOKE_BASELINES, run_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--target-repeats", type=int, default=2)
    ap.add_argument("--attacks", default="black_box,white_box,sybil,coordinated,adaptive")
    ap.add_argument("--baselines", default=",".join(SMOKE_BASELINES))
    ap.add_argument("--seeds", default="20260921")
    ap.add_argument("--scope", default="REAL-DATA RESEARCH MATRIX")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override such as scorer.model_path=/path/model.joblib")
    args = ap.parse_args()
    attacks = [x for x in args.attacks.split(",") if x]
    baselines = [x for x in args.baselines.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    fixture = load_beir_fixture(args.corpus, args.queries, args.qrels,
                                target_limit=args.targets, seed=seeds[0])
    payload = run_matrix(attacks, baselines, seeds, fixture, args.out,
                         config_path=args.config, target_limit=args.targets,
                         repeat_targets=args.target_repeats, scope_label=args.scope,
                         resume=True, overrides=args.overrides)
    print(json.dumps({
        "cases_completed": payload["cases_completed"],
        "cases_requested": payload["cases_requested"],
        "failures": payload["failures"],
        "scope": payload["scope"],
        "output": str(Path(args.out) / "results.json"),
    }, indent=2))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
