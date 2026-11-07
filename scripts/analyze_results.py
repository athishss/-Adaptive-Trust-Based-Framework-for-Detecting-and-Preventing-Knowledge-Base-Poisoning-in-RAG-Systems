#!/usr/bin/env python3
"""Bootstrap CIs and paired exact-McNemar (Holm) over a finished matrix directory.

    python scripts/analyze_results.py runs/poisonedrag_official --reference RAGLedger
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from trace_rag.evaluation.analysis import analyse_directory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--reference", default="RAGLedger")
    ap.add_argument("--scope", default="experiment")
    ap.add_argument("--min-units", type=int, default=10,
                    help="below this many paired units no p-value is reported")
    ap.add_argument("--audit-sample", type=int, default=0, metavar="N",
                    help="also write audit_sample.csv: N answers per stratum for a manual check "
                         "of the ASR criterion (fill the human_label column)")
    a = ap.parse_args()
    payload = analyse_directory(a.results_dir, a.reference, a.scope, a.min_units)
    if a.audit_sample:
        import csv
        from trace_rag.evaluation.analysis import audit_sample
        rows = audit_sample(a.results_dir, per_stratum=a.audit_sample)
        path = Path(a.results_dir) / "audit_sample.csv"
        if rows:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"{len(rows)} answers to label -> {path}")
    print(f"{payload['units']} units -> {a.results_dir}/analysis.md, analysis.json, analysis.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
