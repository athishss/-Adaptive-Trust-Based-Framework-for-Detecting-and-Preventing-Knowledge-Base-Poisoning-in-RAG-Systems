"""Build an experiment :class:`Fixture` from the PoisonedRAG authors' published poison.

Source: ``results/adv_targeted_results/<dataset>.json`` in
https://github.com/sleeepeer/PoisonedRAG (MIT licence).  Each record carries a BEIR
query ``id``, the ``question``, the ``correct answer``, the target ``incorrect answer`` and
five attacker passages (``adv_texts``).  Nothing here generates or rewrites poison; the
attacks ``poisonedrag_official`` / ``poisonedrag_official_sybil`` inject ``adv_texts``
verbatim.

Rules that keep the denominator honest:

* a question whose gold passage is not in the supplied corpus is *dropped and counted*,
  never replaced; a blank corpus passage is skipped and a question whose gold
  passage is blank is dropped separately;
* ``target_limit`` kept questions starting at ``target_offset`` are attacked, the others are background traffic;
* ``max_clean`` shrinks the corpus for cost but always keeps every selected question's
  gold passage; the remainder is a deterministic sample (``seed``);
* the correct answer comes from the authors' file and is evaluator-only.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ingestion.parsers import iter_beir_corpus
from ..ingestion.subset import read_qrels_set
from .experiment import Fixture

SOURCE_NOTE = "PoisonedRAG published adv_targeted_results (unmodified)"


def load_poisonedrag_records(poison_json: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(poison_json).read_text(encoding="utf-8"))
    rows = list(payload.values()) if isinstance(payload, dict) else list(payload)
    if not rows:
        raise ValueError(f"no records in {poison_json}")
    return rows


def load_poisonedrag_fixture(poison_json: str | Path, corpus: str | Path, qrels: str | Path,
                             *, target_limit: Optional[int] = None, target_offset: int = 0,
                             max_clean: Optional[int] = None,
                             seed: int = 20260921,
                             n_sources: int = 20) -> Tuple[Fixture, Dict[str, Any]]:
    """Return ``(fixture, report)``; ``report`` lists every drop and its reason."""
    records = load_poisonedrag_records(poison_json)
    gold_ids = {q: sorted(d) for q, d in read_qrels_set(Path(qrels)).items()}
    wanted = {d for r in records for d in gold_ids.get(str(r.get("id")), [])}

    texts: Dict[str, Dict[str, str]] = {}
    rows: List[Dict[str, str]] = []
    blank_doc_ids: set[str] = set()
    for row in iter_beir_corpus(corpus):
        doc_id = str(row.get("doc_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not text:
            # BEIR corpora can contain title-only/empty rows.  They cannot be
            # ingested under the stream contract and must not consume a clean
            # passage budget; count them instead of failing mid-run.
            blank_doc_ids.add(doc_id)
            continue
        entry = {"_id": doc_id, "title": str(row.get("title") or ""), "text": text}
        rows.append(entry)
        if doc_id in wanted:
            texts[doc_id] = entry

    questions: List[Dict[str, Any]] = []
    false_answers: Dict[str, str] = {}
    dropped = {"incomplete_record": 0, "no_qrels": 0,
               "gold_passage_not_in_corpus": 0, "gold_passage_empty_text": 0}
    for r in records:
        qid = str(r.get("id"))
        adv = [str(t) for t in (r.get("adv_texts") or []) if str(t).strip()]
        correct = str(r.get("correct answer", "")).strip()
        wrong = str(r.get("incorrect answer", "")).strip()
        if not (adv and correct and wrong and r.get("question")):
            dropped["incomplete_record"] += 1
            continue
        if qid not in gold_ids:
            dropped["no_qrels"] += 1
            continue
        doc_id = next((d for d in gold_ids[qid] if d in texts), None)
        if doc_id is None:
            if any(d in blank_doc_ids for d in gold_ids[qid]):
                dropped["gold_passage_empty_text"] += 1
            else:
                dropped["gold_passage_not_in_corpus"] += 1
            continue
        questions.append({"qid": qid, "question": str(r["question"]), "gold_answer": correct,
                          "gold_answers": [correct], "gold_doc_id": doc_id, "adv_texts": adv})
        false_answers[qid] = wrong
    if not questions:
        raise ValueError("no PoisonedRAG question has a gold passage in the corpus; rebuild the "
                         "subset with scripts/download_data.py --include-queries <poison json>")

    # The first ``target_limit`` questions are attacked; the rest are the clean
    # background traffic the stream needs (a background query identical to a target
    # query would break the stream contract, so at least one non-target is required).
    offset = max(0, int(target_offset))
    n_targets = (len(questions) - offset) if not target_limit else min(int(target_limit),
                                                                       len(questions) - offset)
    if n_targets < 1 or offset + n_targets >= len(questions):
        raise ValueError(
            f"{len(questions)} usable question(s) cannot supply {target_limit or 'all'} target(s) "
            f"from offset {offset} plus non-target background questions; lower --targets or "
            "supply more questions")
    keep_gold = {q["gold_doc_id"] for q in questions}
    pool = [r for r in rows if r["_id"] not in keep_gold]
    if max_clean is not None and len(keep_gold) + len(pool) > int(max_clean):
        rng = random.Random(seed)
        pool = rng.sample(pool, max(0, int(max_clean) - len(keep_gold)))
    clean_rows = [texts[d] for d in sorted(keep_gold)] + pool
    clean: List[Dict[str, Any]] = []
    for row in clean_rows:
        digest = hashlib.sha256(row["_id"].encode()).hexdigest()
        clean.append({**row, "source_id": f"beir_source_{int(digest[:8], 16) % int(n_sources):02d}"})

    fixture = Fixture(clean=clean, late_clean=[], questions=questions,
                      targets=[q["qid"] for q in questions[offset:offset + n_targets]],
                      false_answers=false_answers)
    report = {"records": len(records), "kept": len(questions), "targets": n_targets, "target_offset": offset,
              "background_questions": len(questions) - n_targets, "dropped": dropped,
              "blank_passages_skipped": len(blank_doc_ids),
              "clean_passages": len(clean), "adv_texts_per_question":
              sum(len(q["adv_texts"]) for q in questions) / len(questions),
              "source": SOURCE_NOTE}
    return fixture, report


__all__ = ["load_poisonedrag_fixture", "load_poisonedrag_records", "SOURCE_NOTE"]
