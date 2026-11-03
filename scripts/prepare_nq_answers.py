#!/usr/bin/env python3
"""Build evaluator-only NQ gold answers for a BEIR query subset.

NQ Open is a small public question/answer dataset.  We match its questions to
the prepared BEIR queries after case/punctuation normalization.  The output is
JSONL keyed by the BEIR query id, never shown to the running defence.

    python scripts/prepare_nq_answers.py \
      --queries data/nq_pilot/queries_subset.jsonl \
      --out data/nq_pilot/answers.jsonl

The builder fails if any query lacks an answer.  Partial gold-answer coverage
would make the ASR denominator ambiguous, so it is rejected.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HF = ("https://huggingface.co/datasets/google-research-datasets/nq_open/"
      "resolve/main/nq_open/validation-00000-of-00001.parquet")


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def load_queries(path: Path):
    rows = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append({"qid": str(row.get("_id") or row.get("qid")),
                             "text": str(row.get("text") or row.get("question") or "")})
    else:
        import pyarrow.parquet as pq
        for row in pq.read_table(path).to_pylist():
            rows.append({"qid": str(row.get("_id") or row.get("id")),
                         "text": str(row.get("text") or row.get("question") or "")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq-open", default=None,
                    help="local nq_open validation parquet (downloads 213 KB if omitted)")
    args = ap.parse_args()
    query_path, out_path = Path(args.queries), Path(args.out)
    if not query_path.exists():
        raise SystemExit(f"missing queries file: {query_path}")
    source = Path(args.nq_open) if args.nq_open else Path(".cache/nq_open_validation.parquet")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading NQ Open validation -> {source}", flush=True)
        urllib.request.urlretrieve(HF, source)
    import pyarrow.parquet as pq
    nq_rows = pq.read_table(source).to_pylist()
    answer_by_question = {}
    for row in nq_rows:
        answers = row.get("answer") or row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(item).strip() for item in answers if str(item).strip()]
        if answers:
            answer_by_question.setdefault(normalize(row.get("question", "")), answers)
    queries = load_queries(query_path)
    missing = []
    output = []
    for row in queries:
        answers = answer_by_question.get(normalize(row["text"]))
        if not answers:
            missing.append(row["qid"])
            continue
        output.append({"qid": row["qid"], "gold_answer": answers[0],
                       "gold_answers": answers})
    if missing:
        print(f"{len(missing)} of {len(queries)} queries have no NQ Open answer match: "
              + ", ".join(missing[:20]), file=sys.stderr)
        print("Refusing partial ASR coverage. Use a compatible query source or an explicit "
              "answer file; do not fill missing answers with guesses.", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
                        encoding="utf-8")
    print(json.dumps({"queries": len(output), "answers": out_path,
                      "coverage": 1.0, "source": str(source)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
