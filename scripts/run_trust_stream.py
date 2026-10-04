#!/usr/bin/env python3
"""End-to-end trust run over real data (Person B's smoke test).

Runs the whole system on a BEIR corpus with Person B's trust layer wired in,
so a GPU box (Colab, a workstation) can exercise the defence on real passages:

  1. ingest the clean corpus with simulated contributors (Person A's assigner);
  2. index it (the GPU step);
  3. for each target question, take the passage that actually carries the gold
     answer (from BEIR ``qrels.tsv`` when it is there, otherwise the passage in
     the retrieved pool that contains it) and corrupt *that* with Person C's
     attack helpers (negation / entity swap), then ingest the poison as a burst
     from one fresh attacker source;
  4. replay a Zipf-repeated query stream, answering through the full pipeline
     with TrustLedger + CorroborationVerifier + TrustPolicy wired in;
  5. write metrics, the trust history, the B6 figures, and a short summary.

Crafting the poison from the gold passage is what makes the test meaningful:
corrupting an arbitrary top-ranked passage produced attacks the verifier could
not adjudicate, because the clean corroborating text was not about the same
claim.  With the gold passage corrupted, the clean copy of it is still in the
index, so the pool holds an independent passage that contradicts the poison -
exactly the situation the verifier exists for.

This is deliberately *not* the headline experiment: Person C's runner (C6) owns
the V0-V5 ablations, the real baselines and the confidence intervals.  What this
script establishes is that the pieces run together on real data and that the
trust dynamics behave - the numbers it prints are a smoke test, and it says so
in the output.

Usage (Colab / local):

    python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
        --corpus data/nq/corpus_subset.parquet --queries data/nq/queries_subset.jsonl \
        --qrels data/nq/qrels.tsv \
        --steps 40 --targets 3 --out runs/nq/trust_stream

Offline sanity check on the bundled mini corpus:

    python scripts/run_trust_stream.py --config config/default.yaml \
        --corpus examples/mini_corpus.jsonl --queries examples/mini_questions.json \
        --steps 12 --targets 2 --out runs/trust_stream_mini

On Colab with a real model, switch the generation backend without editing the
shared config:

    python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
        --corpus data/nq/corpus_subset.parquet --queries data/nq/queries_subset.jsonl \
        --qrels data/nq/qrels.tsv --out runs/nq/trust_stream \
        --set generation.backend=ollama --set generation.model_name=qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from trace_rag.attacks.poisoned_rag import entity_swap, make_poisoned_document, negation
from trace_rag.attacks.stream import zipf_queries
from trace_rag.cli import apply_overrides
from trace_rag.config import Config
from trace_rag.ingestion import FixedSourceAssigner, Ingestor, SourceAssigner, iter_beir_corpus
from trace_rag.pipeline import PersonAPipeline
from trace_rag.trust import CorroborationVerifier, TrustConfig, TrustLedger, TrustPolicy
from trace_rag.trust.plots import plot_quarantine_timeline, plot_trust_dynamics, write_history_csv
from trace_rag.trust.queue import VerificationQueue

ATTACKER_SOURCE = "attacker_burst"
WRONG_ANSWER = "Zog the Alien"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_questions(path: Path) -> List[Dict[str, str]]:
    """Accept either the mini-corpus question format or a BEIR queries file."""
    text = path.read_text(encoding="utf-8").strip()
    questions: List[Dict[str, str]] = []
    if text.startswith("{"):
        payload = json.loads(text)
        for item in payload.get("questions", []):
            questions.append({
                "qid": str(item.get("qid") or item.get("_id")),
                "question": str(item["question"]),
                "gold_answer": str(item.get("gold_answer", "")),
            })
        return questions
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            questions.append({
                "qid": str(row.get("_id") or row.get("qid")),
                "question": str(row.get("text") or row.get("question")),
                "gold_answer": str(row.get("gold_answer", "")),
            })
    return questions


def first_sentence(text: str) -> str:
    for separator in (". ", "! ", "? "):
        if separator in text:
            return text.split(separator)[0].strip() + "."
    return text.strip()


def corrupt_passage(text: str, gold_answer: str) -> Tuple[str, str]:
    """Return (poison_text, attack_type) built with Person C's helpers."""
    if gold_answer and gold_answer.lower() in text.lower():
        return entity_swap(text, gold_answer, WRONG_ANSWER), "poisonedrag_bbox"
    sentence = first_sentence(text)
    return negation(text, sentence), "corruption_negation"


def load_qrels(path: Path) -> Dict[str, List[str]]:
    """BEIR qrels (``query-id \\t corpus-id \\t score``, header optional).

    Only positive graded relevance counts: those are the passages the dataset
    itself says answer the question.
    """
    gold: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0].strip().lower() in {"query-id", "query_id", "qid"}:
            continue
        try:
            score = float(parts[2])
        except ValueError:
            continue
        if score > 0:
            gold.setdefault(parts[0].strip(), []).append(parts[1].strip())
    return gold


def fetch_passages(corpus_path: Path, wanted: Set[str]) -> Dict[str, str]:
    """Stream the corpus once and return the text of the wanted doc ids.

    The text is rebuilt exactly as the ingestor builds it (title + passage),
    so the poisoned copy differs from the clean one only by the attack.
    """
    found: Dict[str, str] = {}
    for row in iter_beir_corpus(corpus_path):
        doc_id = str(row.get("doc_id") or "")
        if doc_id in wanted:
            found[doc_id] = " ".join(part for part in (row.get("title", ""),
                                                       row.get("text", "")) if part)
            if len(found) >= len(wanted):
                break
    return found


def choose_target(outcome, gold_texts: Dict[str, str], gold_ids: List[str],
                  gold_answer: str) -> Optional[Tuple[str, str, str]]:
    """Pick the passage to corrupt.  Returns (text, origin_doc_id, source).

    Preference order, best first:
      1. a qrels gold passage that actually contains the gold answer;
      2. a retrieved pool passage that contains the gold answer;
      3. the first qrels gold passage anyway;
      4. the top-ranked retrieved passage (negation attack only).
    """
    answer = gold_answer.strip().lower()
    for doc_id in gold_ids:
        text = gold_texts.get(doc_id, "")
        if text and answer and answer in text.lower():
            return text, doc_id, "qrels gold (answer match)"
    if answer:
        for document in outcome.pool:
            if answer in (document.text or "").lower():
                return document.text, document.doc_id, "retrieved pool (answer match)"
    for doc_id in gold_ids:
        if gold_texts.get(doc_id):
            return gold_texts[doc_id], doc_id, "qrels gold (no answer match)"
    if outcome.documents:
        return outcome.documents[0].text, outcome.documents[0].doc_id, "top-1 retrieved"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/nq_gpu.yaml")
    parser.add_argument("--corpus", required=True, help="BEIR corpus (Parquet or JSONL)")
    parser.add_argument("--queries", required=True, help="queries_subset.jsonl or mini questions")
    parser.add_argument("--qrels", default=None,
                        help="BEIR qrels.tsv giving each query's gold passages "
                             "(default: qrels.tsv sitting next to --queries)")
    parser.add_argument("--out", default="runs/trust_stream")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=None,
                        help="override any config value, e.g. "
                             "--set generation.backend=ollama --set retrieval.top_k=5 "
                             "(repeatable; same syntax as the trace-rag CLI)")
    parser.add_argument("--limit", type=int, default=None, help="ingest only N clean passages")
    parser.add_argument("--steps", type=int, default=40, help="queries in the stream")
    parser.add_argument("--targets", type=int, default=3, help="target questions to poison")
    parser.add_argument("--n-sources", type=int, default=2000, help="simulated contributors")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--skip-index", action="store_true",
                        help="reuse the saved index from a previous run")
    parser.add_argument("--force-medium", action="store_true",
                        help="demo mode: escalate every passage so the verifier path "
                             "always runs (the calibrated bands are the real setting)")
    parser.add_argument("--quarantine-refutations", type=int, default=2,
                        help="refutations before a document is quarantined (plan default 2)")
    parser.add_argument("--no-hierarchical", action="store_true",
                        help="V2 ablation: document-only trust, no family/source prior "
                             "and no inheritance")
    parser.add_argument("--max-corroboration", type=int, default=3,
                        help="independent passages the verifier will consult")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = apply_overrides(Config.load(args.config, storage={"root": str(out_dir)}), args.set)
    pipeline = PersonAPipeline.from_config(config, load_existing_index=args.skip_index)
    questions = load_questions(Path(args.queries))
    if not questions:
        print("no questions found", file=sys.stderr)
        return 2
    targets = questions[: max(1, args.targets)]

    # ---------------------------------------------------------------- ingest
    banner("1. Ingest the clean corpus with simulated contributors")
    ingestor = Ingestor(pipeline.store, config.ingestion)
    if not args.skip_index or pipeline.store.counts().get("chunks", 0) == 0:
        report = ingestor.ingest_beir(args.corpus, SourceAssigner(n_sources=args.n_sources,
                                                                 seed=args.seed),
                                      limit=args.limit, seconds_per_doc=60.0)
        print(f"  ingested: {report.to_dict()}")
    print(f"  store: {pipeline.store.counts()}")

    # ---------------------------------------------------------------- index
    banner("2. Index the clean corpus (the GPU step)")
    if not args.skip_index:
        indexed = pipeline.index_chunks()
        pipeline.save_index()
        print(f"  indexed: {indexed} passages   index size: {len(pipeline.index)}")

    # ---------------------------------------------------------------- poison
    banner("3. Craft poison from each target's gold passage (Person C's helpers)")
    qrels_path = Path(args.qrels) if args.qrels else Path(args.queries).with_name("qrels.tsv")
    gold_ids = load_qrels(qrels_path) if qrels_path.exists() else {}
    if gold_ids:
        print(f"  qrels: {qrels_path} ({len(gold_ids)} queries with a gold passage)")
    else:
        print("  no qrels file: targets fall back to the retrieved pool")
    wanted = {doc_id for target in targets for doc_id in gold_ids.get(target["qid"], [])}
    gold_texts = fetch_passages(Path(args.corpus), wanted) if wanted else {}
    poison_docs = []
    poison_chunk_ids: set = set()
    poison_info: List[Dict[str, str]] = []
    for index, target in enumerate(targets):
        outcome = pipeline.retriever.retrieve(target["question"], f"craft_{target['qid']}")
        chosen = choose_target(outcome, gold_texts, gold_ids.get(target["qid"], []),
                               target["gold_answer"])
        if chosen is None:
            print(f"  {target['qid']}: no passage retrieved, skipped")
            continue
        target_text, origin, how = chosen
        poison_text, attack_type = corrupt_passage(target_text, target["gold_answer"])
        record = make_poisoned_document(
            doc_id=f"poison_{target['qid']}", chunk_id=f"poison_{target['qid']}#0000",
            text=poison_text, source_id=ATTACKER_SOURCE,
            family_id=f"poison_family_{index}", attack_type=attack_type,
            target_query=target["question"],
            metadata={"source_of_truth": origin},
        )
        chunks = ingestor.ingest_text(record.doc_id, record.text, record.source_id,
                                      ingested_at=1_800_000_000.0 + index * 20.0,
                                      passage_mode=True)
        # The pipeline and the ledger work in chunk ids (``doc#0000``), so keep
        # those: comparing document ids against them silently matched nothing.
        poison_chunk_ids.update(chunk.chunk_id for chunk in chunks)
        poison_docs.append(record)
        poison_info.append({"doc_id": record.doc_id, "origin": origin, "chosen_by": how,
                            "attack": attack_type, "chunk_ids": [c.chunk_id for c in chunks]})
        print(f"  {record.doc_id}: {attack_type} on {origin} [{how}] "
              f"-> {[c.chunk_id for c in chunks]}")
    if not poison_docs:
        print("no poison was generated - nothing to test", file=sys.stderr)
        return 2
    pipeline.index_chunks()
    pipeline.save_index()
    print(f"  index size after poison: {len(pipeline.index)}")

    # ---------------------------------------------------------------- trust
    banner("4. Wire in Person B's trust layer")
    trust_config = TrustConfig(quarantine_refutations=args.quarantine_refutations,
                               hierarchical=not args.no_hierarchical)
    ledger = TrustLedger(out_dir / "trust.sqlite3", store=pipeline.store, config=trust_config)
    verifier = CorroborationVerifier(llm=pipeline.generator.llm,
                                     max_corroboration=args.max_corroboration)
    queue = VerificationQueue()
    policy = TrustPolicy(ledger=ledger, on_quarantine=pipeline.on_quarantine, queue=queue)
    pipeline.trust_provider = ledger
    pipeline.retriever.trust = ledger
    pipeline.verifier = verifier
    pipeline.policy = policy
    if args.force_medium:
        # Demo mode, labelled as such in the metrics: every passage escalates so
        # the verifier path is exercised even when the heuristic scorer is calm.
        pipeline.scorer.theta_low = 0.0
        pipeline.scorer.theta_high = 1.01
    print(f"    verifier mode: {verifier.mode}   "
          f"(NLI runs on {config.embedding.device} embeddings, CUDA when visible)")
    print(f"    quarantine after {args.quarantine_refutations} refutation(s); "
          f"hierarchical={trust_config.hierarchical}")
    if args.force_medium:
        print("    demo mode: every passage escalated to MEDIUM (--force-medium)")

    # ---------------------------------------------------------------- stream
    banner(f"5. Replay the query stream ({args.steps} steps, Zipf repetition)")
    stream = zipf_queries([q["question"] for q in questions], args.steps, seed=args.seed)
    target_text = {q["question"].strip().lower() for q in targets}

    per_step: List[Dict[str, object]] = []
    poison_ids = poison_chunk_ids
    for step, question in enumerate(stream):
        result = pipeline.answer(question, f"stream_{step:04d}")
        drained = policy.drain_verification_queue(verifier)
        cited_poison = [d.doc_id for d in result.retrieval.documents
                        if d.doc_id in poison_ids and d.doc_id in (result.answer or "")]
        retrieved_ids = [d.doc_id for d in result.retrieval.documents]
        per_step.append({
            "step": step,
            "query": question,
            "is_target": question.strip().lower() in target_text,
            "answer": result.answer,
            "abstained": bool(result.record.abstained),
            "retrieved": retrieved_ids,
            "poison_retrieved": any(doc_id in poison_ids for doc_id in retrieved_ids),
            "citations": [d.doc_id for d in result.retrieval.documents
                          if d.doc_id in (result.answer or "")],
            "poison_cited": bool(cited_poison),
            "bands": {band: sum(1 for a in result.assessments if a.band.value == band)
                      for band in ("LOW", "MEDIUM", "HIGH")},
            "verified": [v.outcome.value for v in result.decision.verified],
            "verified_docs": [[v.doc_id, v.outcome.value] for v in result.decision.verified],
            "drained_verified": [v.outcome.value for v in drained],
            "newly_quarantined": list(result.decision.notes.get("newly_quarantined", [])),
            "llm_calls": int(result.llm_calls),
            "latency_ms": float(result.timings_ms.get("total", 0.0)),
        })

    # ---------------------------------------------------------------- metrics
    banner("6. Smoke-test metrics (not the headline evaluation)")
    target_steps = [row for row in per_step if row["is_target"]]
    poison_answers = [row for row in target_steps if row["poison_cited"]]
    poison_retrieved = [row for row in target_steps if row["poison_retrieved"]]

    trust_snaps = ledger.get_trust(sorted(poison_chunk_ids))
    blocked = ledger.blocked_doc_ids()
    poisoned_blocked = sorted(blocked & poison_ids)
    clean_blocked = sorted(blocked - poison_ids)
    first_quarantine = next((row["step"] for row in per_step if row["newly_quarantined"]), None)

    metrics = {
        "smoke_test": True,
        "note": "component smoke test; Person C's runner owns ASR, baselines and CIs",
        "bands": "forced MEDIUM (demo mode)" if args.force_medium else "calibrated/heuristic",
        "config": str(args.config),
        "corpus": str(args.corpus),
        "steps": len(per_step),
        "target_steps": len(target_steps),
        "verifier_mode": verifier.mode,
        "llm": ("stub (claim extraction is lexical: verdicts are indicative only)"
                if getattr(pipeline.generator.llm, "name", "") == "stub"
                else getattr(pipeline.generator.llm, "name", "unknown")),
        "poison_documents": len(poison_docs),
        "poison_targets": poison_info,
        "poison_retrieved_rate": (len(poison_retrieved) / len(target_steps)) if target_steps else 0.0,
        "poison_cited_in_answer": len(poison_answers),
        "poison_citation_rate": (len(poison_answers) / len(target_steps)) if target_steps else 0.0,
        "poison_quarantined": len(poisoned_blocked),
        "clean_false_quarantine": len(clean_blocked),
        "clean_false_quarantine_ids": clean_blocked,
        "first_quarantine_step": first_quarantine,
        "mean_llm_calls_per_query": round(statistics.fmean(
            [row["llm_calls"] for row in per_step]) if per_step else 0.0, 3),
        "mean_latency_ms": round(statistics.fmean(
            [row["latency_ms"] for row in per_step]) if per_step else 0.0, 2),
        "verifications": {
            "SUPPORT": sum(row["verified"].count("SUPPORT") for row in per_step),
            "REFUTE": sum(row["verified"].count("REFUTE") for row in per_step),
            "NEUTRAL": sum(row["verified"].count("NEUTRAL") for row in per_step),
        },
        "queued_verifications": sum(len(row["drained_verified"]) for row in per_step),
        "poison_trust": {doc_id: {"t_eff": round(snap.t_eff, 4), "status": snap.status.value}
                         for doc_id, snap in trust_snaps.items()},
    }
    print(json.dumps(metrics, indent=2))

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "stream.jsonl").write_text(
        "\n".join(json.dumps(row, default=str) for row in per_step), encoding="utf-8")

    # ---------------------------------------------------------------- figures
    banner("7. Trust-dynamics figures (B6)")
    history = ledger.export_history()
    write_history_csv(history, out_dir / "trust_history.csv")
    figures = plot_trust_dynamics(history, out_dir / "figures")
    timeline = plot_quarantine_timeline(ledger.get_audit_log(limit=10000),
                                        out_dir / "figures" / "quarantine_timeline.png")
    print(f"  history rows: {len(history)}   figures: {len(figures)}")
    for path in figures:
        print(f"    {path}")
    if timeline:
        print(f"    {timeline}")

    ledger.close()
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
