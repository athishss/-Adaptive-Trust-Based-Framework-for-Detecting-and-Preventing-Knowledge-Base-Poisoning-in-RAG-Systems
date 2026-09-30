#!/usr/bin/env python3
"""Work package A7: latency profile and signal ablation.

Two questions this answers for the report:

  1. Where does per-query time go (retrieval / signals / policy / generation),
     mean and p95, and how many LLM calls per query?
  2. Which of the six signals actually matter?  Each signal is switched off in
     turn, the scorer is retrained on the remaining features, and the drop in
     held-out ROC AUC is reported.

Labels come from the evaluator-only side (Person C).  Pass ``--poison-prefix``
for the bundled demo corpus, or ``--labels labels.json`` with a list of poisoned
passage ids for a real run.

    python scripts/profile_and_ablate.py --out runs/a7_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from trace_rag import Config, PersonAPipeline
from trace_rag.contracts import SIGNAL_NAMES
from trace_rag.detection import TrainingSet, build_rows, split_by_question, train_scorer
from trace_rag.detection.scorer import SuspicionScorer
from trace_rag.detection.training import evaluate
from trace_rag.ingestion import Ingestor

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_demo_pipeline(config: Config) -> PersonAPipeline:
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    ingestor = Ingestor(pipeline.store, config.ingestion)
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_corpus.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=1_600_000_000.0 + i * 86_400.0, passage_mode=True)
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_late_clean.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=1_750_000_000.0 + i * 45.0, passage_mode=True)
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_poison.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=1_750_000_000.0 + i * 30.0, passage_mode=True)
    pipeline.index_chunks()
    return pipeline


def collect_dataset(pipeline: PersonAPipeline, questions: Sequence[dict],
                    is_poison: Callable[[str], bool]) -> TrainingSet:
    dataset = TrainingSet()
    for question in questions:
        for suffix in ("", " ?"):
            query = question["question"] + suffix
            query_id = f"{question['qid']}{suffix}"
            outcome = pipeline.retriever.retrieve(query, query_id)
            snapshots = pipeline.signals.compute(query, outcome.documents, outcome.pool)
            dataset.extend(build_rows(query_id, outcome.documents, snapshots, is_poison,
                                      attack_family="poisonedrag_bb"))
    return dataset


def latency_profile(pipeline: PersonAPipeline, questions: Sequence[dict], repeats: int = 3
                    ) -> Dict[str, Dict[str, float]]:
    samples: Dict[str, List[float]] = {}
    llm_calls: List[int] = []
    for _ in range(repeats):
        for question in questions:
            result = pipeline.answer(question["question"], f"prof_{question['qid']}", log=False)
            llm_calls.append(result.llm_calls)
            for stage, value in result.timings_ms.items():
                samples.setdefault(stage, []).append(value)
    profile = {}
    for stage, values in samples.items():
        ordered = sorted(values)
        profile[stage] = {
            "mean_ms": round(stats.mean(values), 3),
            "p50_ms": round(ordered[len(ordered) // 2], 3),
            "p95_ms": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 3),
            "n": len(values),
        }
    profile["llm_calls_per_query"] = {"mean": round(stats.mean(llm_calls), 3),
                                      "max": max(llm_calls), "n": len(llm_calls)}
    return profile


def signal_ablation(dataset: TrainingSet, config: Config) -> Dict[str, Dict[str, float]]:
    """Retrain with each signal zeroed in turn; report held-out AUC and the drop."""
    results: Dict[str, Dict[str, float]] = {}
    _, _, held_out = train_scorer(dataset, config.scorer)
    baseline_auc = float(held_out.get("roc_auc", float("nan")))
    results["all_signals"] = {"roc_auc": round(baseline_auc, 4), "delta": 0.0,
                             "high_band_recall": round(float(held_out.get("high_band_recall", 0)), 4)}

    features = list(dataset.feature_names)
    train_idx, val_idx, test_idx = split_by_question(dataset.query_ids, seed=config.scorer.random_state)
    X, y = dataset.X, dataset.y
    for signal in SIGNAL_NAMES:
        column = features.index(signal)
        X_ablated = X.copy()
        X_ablated[:, column] = 0.0
        scorer = SuspicionScorer(feature_names=features, C=config.scorer.C,
                                 calibration=config.scorer.calibration,
                                 random_state=config.scorer.random_state)
        scorer.fit(X_ablated[train_idx], y[train_idx])
        if config.scorer.calibration != "none":
            scorer.calibrate(X_ablated[val_idx], y[val_idx])
        scorer.select_thresholds(scorer.predict_proba(X_ablated[val_idx]), y[val_idx],
                                 [dataset.query_ids[i] for i in val_idx],
                                 config.scorer.target_high_fpr, config.scorer.escalation_budget)
        metrics = evaluate(scorer, X_ablated[test_idx], y[test_idx],
                           [dataset.query_ids[i] for i in test_idx])
        auc = float(metrics.get("roc_auc", float("nan")))
        results[f"without_{signal}"] = {
            "roc_auc": round(auc, 4),
            "delta": round(auc - baseline_auc, 4),
            "high_band_recall": round(float(metrics.get("high_band_recall", 0.0)), 4),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default="runs/a7")
    parser.add_argument("--poison-prefix", default="poison",
                        help="passage ids starting with this are poisoned (demo corpus)")
    parser.add_argument("--labels", default=None, help="JSON list of poisoned passage ids")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = Config.load(args.config) if args.config else Config()
    config.storage.root = args.root
    pipeline = build_demo_pipeline(config)
    questions = json.loads((EXAMPLES / "mini_questions.json").read_text(encoding="utf-8"))["questions"]

    if args.labels:
        poison_ids = set(json.loads(Path(args.labels).read_text(encoding="utf-8")))
        is_poison = lambda doc_id: doc_id in poison_ids            # noqa: E731
    else:
        is_poison = lambda doc_id: doc_id.startswith(args.poison_prefix)   # noqa: E731

    report = {
        "latency": latency_profile(pipeline, questions, args.repeats),
        "ablation": signal_ablation(collect_dataset(pipeline, questions, is_poison), config),
        "pipeline": pipeline.stats(),
        "caveat": ("Latency excludes real embedding and LLM time (offline backends). "
                   "The bundled corpus is trivially separable, so ablation deltas here are "
                   "illustrative; rerun on NQ with Person C's attacks for reportable numbers."),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}")
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
