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
    a = ap.parse_args()
    payload = analyse_directory(a.results_dir, a.reference, a.scope, a.min_units)
    print(f"{payload['units']} units -> {a.results_dir}/analysis.md, analysis.json, analysis.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
