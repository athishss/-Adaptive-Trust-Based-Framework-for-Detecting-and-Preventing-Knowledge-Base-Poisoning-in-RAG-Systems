#!/usr/bin/env python3
"""End-to-end demonstration with Person B's trust components wired in.

Extends the original demo_end_to_end.py by replacing the stand-in DemoLedger
with Person B's real TrustLedger, CorroborationVerifier, and TrustPolicy.

Five stages:
  1. Ingest clean passages, then a burst of PoisonedRAG-style passages;
  2. Answer with the untrained heuristic scorer (no trust yet);
  3. Train the calibrated suspicion scorer;
  4. Answer again with the trained scorer + Person B's full trust framework;
  5. Show how repeated queries lead to automatic quarantine.

Usage:  python scripts/demo_person_b.py [--root runs/demo_b]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from trace_rag import Config, PersonAPipeline
from trace_rag.contracts import TrustStatus
from trace_rag.detection import TrainingSet, build_rows, train_scorer
from trace_rag.ingestion import Ingestor
from trace_rag.trust import TrustLedger, TrustConfig, CorroborationVerifier, TrustPolicy

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CLEAN_START = 1_600_000_000.0
POISON_START = 1_750_000_000.0


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_assessments(result) -> None:
    for a in sorted(result.assessments, key=lambda x: -x.suspicion):
        flag = {"LOW": "   ", "MEDIUM": " ? ", "HIGH": " ! "}[a.band.value]
        label = "POISON" if a.doc_id.startswith("poison") else "clean "
        print(f"  {flag} {label} {a.doc_id:<26} suspicion={a.suspicion:.3f} {a.band.value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/demo_b")
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
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_late_clean.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=POISON_START + i * 45.0, passage_mode=True)
    for i, row in enumerate(load_jsonl(EXAMPLES / "mini_poison.jsonl")):
        ingestor.ingest_text(row["_id"], row["text"], row["source_id"],
                             ingested_at=POISON_START + i * 30.0, passage_mode=True)
    pipeline.index_chunks()
    print(f"  store: {pipeline.store.counts()}   index: {len(pipeline.index)} passages")

    # ---------------------------------------------------------------- stage 2
    banner(f"2. Answer with untrained scorer (no Person B yet)\n   Q: {target_question['question']}")
    before = pipeline.answer(target_question["question"], "demo_before")
    print(f"  A: {before.answer}")
    show_assessments(before)

    # ---------------------------------------------------------------- stage 3
    banner("3. Train the calibrated suspicion scorer")
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
    scorer.save(config.path(config.storage.scorer_path))
    pipeline.scorer = scorer

    # ---------------------------------------------------------------- stage 4
    banner("4. Wire in Person B's trust framework and answer again")

    # Create Person B components
    trust_config = TrustConfig(
        quarantine_refutations=2,    # quarantine after 2 refutations
        reject_refutations=3,
    )
    trust_db = Path(args.root) / "trust.sqlite3"
    ledger = TrustLedger(trust_db, store=pipeline.store, config=trust_config)
    verifier = CorroborationVerifier(llm=pipeline.generator.llm, max_corroboration=3)
    policy = TrustPolicy(ledger=ledger, on_quarantine=pipeline.on_quarantine)

    # Wire them in
    pipeline.trust_provider = ledger
    pipeline.retriever.trust = ledger
    pipeline.verifier = verifier
    pipeline.policy = policy

    print("  Person B components wired in:")
    print(f"    TrustLedger:  {trust_db}")
    print(f"    Verifier:     CorroborationVerifier (max_corroboration=3)")
    print(f"    Policy:       TrustPolicy (quarantine_refutations=2)")

    after = pipeline.answer(target_question["question"], "demo_after_b")
    print(f"\n  Q: {target_question['question']}")
    print(f"  A: {after.answer}")
    show_assessments(after)

    if after.decision.verified:
        print(f"\n  Verification results:")
        for v in after.decision.verified:
            print(f"    {v.doc_id}: {v.outcome.value} "
                  f"(support={v.support_mass:.3f} refute={v.refute_mass:.3f} "
                  f"llm_calls={v.llm_calls})")

    excluded = list(after.decision.excluded_doc_ids)
    print(f"  excluded from answer: {excluded or 'none'}")

    # ---------------------------------------------------------------- stage 5
    banner("5. Repeated queries -> automatic quarantine via trust decay")

    # Run the same query multiple times to accumulate refutations
    for i in range(3):
        result = pipeline.answer(target_question["question"], f"demo_repeat_{i}")
        newly_q = result.decision.notes.get("newly_quarantined", [])
        if newly_q:
            print(f"  Query {i+1}: quarantined {newly_q}")

    # Show final trust state of poison passages
    poison_ids = [f"poison_eiffel_{i}#0000" for i in range(3)]
    trust_snaps = ledger.get_trust(poison_ids)
    print(f"\n  Trust state after repeated queries:")
    for doc_id in poison_ids:
        snap = trust_snaps.get(doc_id)
        if snap:
            print(f"    {doc_id}: t_eff={snap.t_eff:.3f} status={snap.status.value} "
                  f"(n_obs={snap.n_doc_observations})")
        else:
            print(f"    {doc_id}: not yet observed")

    blocked = ledger.blocked_doc_ids()
    print(f"\n  Blocked doc_ids: {blocked or 'none'}")

    # Final answer after trust has evolved
    final = pipeline.answer(target_question["question"], "demo_final_b")
    print(f"\n  Final answer (after trust evolution):")
    print(f"  Q: {target_question['question']}")
    print(f"  A: {final.answer}")
    poison_in_final = [d.doc_id for d in final.retrieval.documents
                       if d.doc_id.startswith("poison")]
    print(f"  Poisoned passages in retrieval: {poison_in_final or 'none'}")

    banner("Summary")
    stats = pipeline.stats()
    stats["trust_provider"] = "TrustLedger"
    stats["verifier"] = "CorroborationVerifier"
    stats["policy"] = "TrustPolicy"
    print(json.dumps(stats, indent=2))

    ledger.close()
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
