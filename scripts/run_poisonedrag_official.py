#!/usr/bin/env python3
"""Run the chronological matrix on the PoisonedRAG authors' *published* poison.

    git clone --depth 1 https://github.com/sleeepeer/PoisonedRAG.git external/PoisonedRAG
    python scripts/download_data.py --dataset nq --out data/nq --subset 100000 --queries 500 \
        --include-queries external/PoisonedRAG/results/adv_targeted_results/nq.json
    python scripts/run_poisonedrag_official.py \
        --poison external/PoisonedRAG/results/adv_targeted_results/nq.json \
        --corpus data/nq/corpus_subset.parquet --qrels data/nq/qrels.tsv \
        --config config/nq_gpu.yaml --out runs/poisonedrag_official --targets 20 --max-clean 5000

Both attack variants inject the authors' passages unmodified; they differ only in who
publishes them (one new contributor, or five).  Questions whose gold passage is missing from
the corpus are reported and dropped, never replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from trace_rag.evaluation.experiment import OFFICIAL_ATTACKS, SMOKE_BASELINES, run_matrix  # noqa: E402
from trace_rag.evaluation.poisonedrag import load_poisonedrag_fixture  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison", required=True, help="PoisonedRAG results/adv_targeted_results/<dataset>.json")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--max-clean", type=int, default=None,
                    help="cap the clean corpus (gold passages are always kept); default: use all of it")
    ap.add_argument("--target-repeats", type=int, default=2)
    ap.add_argument("--attacks", default=",".join(OFFICIAL_ATTACKS))
    ap.add_argument("--baselines", default=",".join(SMOKE_BASELINES))
    ap.add_argument("--seeds", default="20260921")
    ap.add_argument("--scope", default="REAL-DATA: published PoisonedRAG poison")
    ap.add_argument("--trustrag-dir", default=None,
                    help="checkout of github.com/HuichiZhou/TrustRAG; required if --baselines "
                         "includes trustrag_official")
    ap.add_argument("--target-offset", type=int, default=0)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x]
    attacks = [x for x in args.attacks.split(",") if x]
    baselines = [x for x in args.baselines.split(",") if x]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Each seed attacks a DIFFERENT, disjoint block of published questions (seed i uses
    # questions [offset + i*targets, offset + (i+1)*targets)).  Seeds therefore give
    # independent samples of questions, which is what the confidence intervals and the
    # paired tests assume.  The same question is never reused across seeds.
    failures, completed, requested, summaries = [], 0, 0, []
    for i, seed in enumerate(seeds):
        offset = args.target_offset + i * args.targets
        fixture, report = load_poisonedrag_fixture(
            args.poison, args.corpus, args.qrels, target_limit=args.targets,
            target_offset=offset, max_clean=args.max_clean, seed=seed)
        seed_out = out if len(seeds) == 1 else out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        (seed_out / "fixture_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, **report}, indent=2))
        payload = run_matrix(attacks, baselines, [seed], fixture, seed_out,
                             config_path=args.config, target_limit=args.targets,
                             repeat_targets=args.target_repeats, scope_label=args.scope,
                             resume=True, overrides=args.overrides,
                             trustrag_dir=args.trustrag_dir)
        completed += payload["cases_completed"]
        requested += payload["cases_requested"]
        failures += payload["failures"]
        summaries.append({"seed": seed, "target_offset": offset, "output": str(seed_out)})
    (out / "run_summary.json").write_text(json.dumps(
        {"seeds": summaries, "cases_completed": completed, "cases_requested": requested,
         "failures": failures}, indent=2), encoding="utf-8")
    print(json.dumps({"cases_completed": completed, "cases_requested": requested,
                      "failures": failures, "output": str(out)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
