#!/usr/bin/env python3
"""Ingest NELA-GT news under real outlets/timestamps and print per-outlet trust-relevant statistics.

    python scripts/ingest_nela.py --nela /path/to/nela-gt-2020.db --labels /path/to/source_labels.csv \
        --out runs/nela --max-docs 50000 --per-source-limit 300

Data is NOT downloaded by this repo.  See ``trace_rag/ingestion/nela.py`` for the assumed formats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from trace_rag.config import Config
from trace_rag.ingestion import Ingestor, ProvenanceStore
from trace_rag.ingestion.nela import ingest_nela, load_source_labels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nela", required=True, help="NELA-GT SQLite file or directory of per-outlet JSON files")
    p.add_argument("--labels", default=None, help="CSV: source,label (reliable|mixed|unreliable)")
    p.add_argument("--int-labels", default=None, help='if the CSV uses integers: "0=reliable,1=mixed,2=unreliable"')
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--per-source-limit", type=int, default=None)
    p.add_argument("--only-labelled", action="store_true")
    a = p.parse_args()

    config = Config.load(a.config, storage={"root": a.out})
    int_map = None
    if a.int_labels:
        int_map = {int(k): v for k, v in (kv.split("=") for kv in a.int_labels.split(","))}
    labels = load_source_labels(a.labels, int_map) if a.labels else None
    store = ProvenanceStore(config.path(config.storage.provenance_db))
    report = ingest_nela(Ingestor(store, config.ingestion), a.nela, labels, a.max_docs,
                         a.per_source_limit, only_labelled=a.only_labelled)
    report.pop("per_source")
    print(json.dumps({**report, "store": store.counts()}, indent=2))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
