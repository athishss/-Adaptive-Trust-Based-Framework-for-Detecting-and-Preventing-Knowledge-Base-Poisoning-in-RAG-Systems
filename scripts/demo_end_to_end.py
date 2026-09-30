#!/usr/bin/env python3
"""End-to-end demonstration of the Person A pipeline on the bundled mini corpus.

Runs offline (hashing embedder + stub LLM), so it works on any machine with no
downloads.  Five stages:

  1. ingest clean passages, then a later batch of PoisonedRAG-style passages;
  2. answer a targeted question with the untrained heuristic scorer;
  3. build labelled rows and train the calibrated suspicion scorer
     (split by question, thresholds from validation only);
  4. re-answer with the trained scorer, showing the bands;
  5. quarantine the injected passages through a stand-in trust ledger and show
     retroactive remediation of the answers already served.

Usage:  python scripts/demo_end_to_end.py [--root runs/demo]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from trace_rag import Config, PersonAPipeline
from trace_rag.contracts import TrustSnapshot, TrustStatus
from trace_rag.detection import TrainingSet, build_rows, train_scorer
from trace_rag.ingestion import Ingestor

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CLEAN_START = 1_600_000_000.0          # clean corpus arrives over months
POISON_START = 1_750_000_000.0         # attack arrives as a late burst


class DemoLedger:
    """Stand-in for Person B's trust ledger (same protocol, fixed values)."""

    def __init__(self) -> None:
        self.values: Dict[str, float] = {}
        self.blocked: set = set()

    def quarantine(self, doc_ids: List[str]) -> None:
        for doc_id in doc_ids:
            self.values[doc_id] = 0.05
            self.blocked.add(doc_id)

    def get_trust(self, doc_ids):
        out = {}
        for doc_id in doc_ids:
            value = self.values.get(doc_id, 0.6)
            status = TrustStatus.QUARANTINED if doc_id in self.blocked else TrustStatus.TRUSTED
            out[doc_id] = TrustSnapshot(doc_id, "s", "f", value, value, value, value, status)
        return out

    def blocked_doc_ids(self):
        return set(self.blocked)


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_assessments(result) -> None:
    for assessment in sorted(result.assessments, key=lambda a: -a.suspicion):
        flag = {"LOW": "   ", "MEDIUM": " ? ", "HIGH": " ! "}[assessment.band.value]
        label = "POISON" if assessment.doc_id.startswith("poison") else "clean "
        print(f"  {flag} {label} {assessment.doc_id:<26} suspicion={assessment.suspicion:.3f} "
              f"{assessment.band.value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/demo")
    args = parser.parse_args()

    config = Config.load(None, storage={"root": args.root})
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    questions = json.loads((EXAMPLES / "mini_questions.json").read_text(encoding="utf-8"))
    target_question = next(q for q in questions["questions"] if q["qid"] == "eiffel")

    # ---------------------------------------------------------------- stage 1
    banner("1. Ingest: clean corpus first, attacker batch later")
    ingestor = Ingestor(pipeline.store, config.ingestion)
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_corpus.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=CLEAN_START + i * 86_400.0, passage_mode=True)
    # Leakage guard (plan Section 6.1): benign contributors also arrive late, from
    # brand-new sources.  Without this, "new source" would mean "poison" by
    # construction and the source-history signals would look perfect for free.
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_late_clean.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=POISON_START + i * 45.0, passage_mode=True)
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_poison.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=POISON_START + i * 30.0, passage_mode=True)
    pipeline.index_chunks()
    print(f"  store: {pipeline.store.counts()}   index: {len(pipeline.index)} passages")
    print("  leakage guard on: benign new contributors arrive in the same late window as the attack")

    # ---------------------------------------------------------------- stage 2
    banner(f"2. Answer with the untrained heuristic scorer\n   Q: {target_question['question']}")
    before = pipeline.answer(target_question["question"], "demo_before")
    print(f"  A: {before.answer}")
    show_assessments(before)

    # ---------------------------------------------------------------- stage 3
    banner("3. Train the calibrated suspicion scorer (labels from the attack harness)")
    dataset = TrainingSet()
    for question in questions["questions"]:
        for variant, suffix in ((question["question"], ""), (question["question"].lower(), " ?")):
            query_id = f"{question['qid']}{suffix}"
            outcome = pipeline.retriever.retrieve(variant + suffix, query_id)
            snapshots = pipeline.signals.compute(variant, outcome.documents, outcome.pool)
            dataset.extend(build_rows(
                query_id=query_id, documents=outcome.documents, snapshots=snapshots,
                is_poison=lambda doc_id: doc_id.startswith("poison"),
                attack_family="poisonedrag_bb", query_time=0.0, captured_at=0.0,
            ))
    print(f"  dataset: {dataset.summary()}")
    scorer, thresholds, held_out = train_scorer(dataset, config.scorer)
    print(f"  thresholds: theta_low={thresholds.theta_low:.3f} theta_high={thresholds.theta_high:.3f}")
    print(f"  validation: HIGH-band FPR={thresholds.achieved_high_fpr:.4f} "
          f"(cap {config.scorer.target_high_fpr}), "
          f"escalations/query={thresholds.achieved_escalations_per_query:.2f} "
          f"(budget {config.scorer.escalation_budget})")
    print(f"  held-out test: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in held_out.items()})}")
    top = sorted(scorer.coefficients().items(), key=lambda kv: -abs(kv[1]))[:4]
    print("  strongest features: " + ", ".join(f"{k}={v:+.2f}" for k, v in top))
    print("  NOTE: this toy corpus is trivially separable (one attack family, 3 near-identical\n"
          "        passages per target), so perfect scores here mean nothing. Reportable numbers\n"
          "        come from NQ + the PoisonedRAG harness with leave-one-attack-family-out.")
    scorer.save(config.path(config.storage.scorer_path))
    pipeline.scorer = scorer

    # ---------------------------------------------------------------- stage 4
    banner("4. Answer again with the trained scorer")
    after = pipeline.answer(target_question["question"], "demo_after")
    print(f"  A: {after.answer}")
    show_assessments(after)
    excluded = [a.doc_id for a in after.assessments if a.band.value == "HIGH"]
    print(f"  excluded from the answer this turn: {excluded or 'none'}")

    # ---------------------------------------------------------------- stage 5
    banner("5. Person B quarantines the injected passages -> retroactive remediation")
    poison_ids = [f"poison_eiffel_{i}#0000" for i in range(3)]
    ledger = DemoLedger()
    ledger.quarantine(poison_ids)
    pipeline.retriever.trust = ledger
    report = pipeline.on_quarantine(poison_ids, reason="refuted by corroboration-gated verifier")
    print(f"  quarantined: {poison_ids}")
    print(f"  answers flagged for review: {report.newly_flagged} "
          f"(exposure window: {report.exposure_window} answers served before quarantine)")
    final = pipeline.answer(target_question["question"], "demo_final")
    print(f"  A (after quarantine): {final.answer}")
    print(f"  poisoned passages retrieved now: "
          f"{[d.doc_id for d in final.retrieval.documents if d.doc_id.startswith('poison')] or 'none'}")

    banner("Summary")
    print(json.dumps(pipeline.stats(), indent=2))
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
