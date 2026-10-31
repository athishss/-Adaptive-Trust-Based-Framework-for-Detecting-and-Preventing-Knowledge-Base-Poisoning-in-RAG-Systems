#!/usr/bin/env python3
"""Run the deterministic chronological smoke matrix (attacks x baselines).

This is the supported end-to-end evaluation entry point that works from a clean
checkout with no model downloads and no GPU:

    python scripts/run_chronological_eval.py --out runs/smoke_matrix

It builds the bundled mini corpus, injects one of several attack families, runs
the stream through the real retrieval -> signals -> policy -> verification ->
trust -> generation -> remediation path, and executes every advertised baseline
on the same fixture and schedule.

Outputs (under ``--out``):

    results.json        every case, with full per-query metrics
    results.csv         one row per (attack, baseline, seed)
    summary.md          generated table + explicit scope statement
    cases/<case>/       metrics.json, stream.jsonl, plan.json, run_config.json

Scope: mini corpus + hashing embedder + stub LLM + lexical verifier unless the
config enables the NLI model.  These are smoke-test results, not benchmark
numbers.  For the real-model Natural Questions protocol see
``docs/REAL_MODEL_EVAL.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trace_rag.evaluation.experiment import (  # noqa: E402
    SMOKE_ATTACKS,
    SMOKE_BASELINES,
    load_fixture,
    run_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--out", default="runs/smoke_matrix")
    parser.add_argument("--examples", default=str(ROOT / "examples"))
    parser.add_argument("--attacks", default=",".join(SMOKE_ATTACKS),
                        help="comma-separated attack names")
    parser.add_argument("--baselines", default=",".join(SMOKE_BASELINES),
                        help="comma-separated baseline names")
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--target-repeats", type=int, default=2,
                        help="times each target query is asked (2 reaches the "
                             "production two-refutation quarantine threshold)")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    fixture = load_fixture(args.examples)
    payload = run_matrix(
        attacks=[item for item in args.attacks.split(",") if item],
        baselines=[item for item in args.baselines.split(",") if item],
        seeds=[args.seed],
        fixture=fixture,
        output_root=args.out,
        config_path=args.config,
        target_limit=args.targets,
        repeat_targets=args.target_repeats,
    )
    print(json.dumps({
        "scope": payload["scope"],
        "cases_completed": payload["cases_completed"],
        "cases_requested": payload["cases_requested"],
        "failures": payload["failures"],
        "output": str(Path(args.out) / "results.json"),
    }, indent=2))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
