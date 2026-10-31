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
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from trace_rag.attacks.poisoned_rag import entity_swap, make_poisoned_document, negation
from trace_rag.attacks.stream import zipf_queries
from trace_rag.cli import apply_overrides
from trace_rag.config import Config
from trace_rag.evaluation.metrics import answer_level_asr_fields
from trace_rag.ingestion import FixedSourceAssigner, Ingestor, SourceAssigner, iter_beir_corpus
from trace_rag.ingestion.subset import verify_subset
from trace_rag.snapshots import backup_snapshot, restore_latest_snapshot
from trace_rag.pipeline import PersonAPipeline
from trace_rag.trust import TrustLedger, TrustPolicy
from trace_rag.trust.plots import plot_quarantine_timeline, plot_trust_dynamics, write_history_csv

ATTACKER_SOURCE = "attacker_burst"
WRONG_ANSWER = "Zog the Alien"
ROOT = Path(__file__).resolve().parent.parent
RUN_PURPOSES = ("smoke", "pilot", "research", "research_lifecycle")


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_query_stream(questions: Sequence[Dict[str, str]], steps: int, seed: int,
                       sample_mode: str) -> List[str]:
    """Build the query sequence and make its sampling semantics explicit.

    ``zipf`` is the trust-lifecycle stream and samples with replacement, so the
    number of steps is not the number of distinct queries.  ``no-repeat``
    samples query *indices* without replacement and refuses to silently shorten
    the request when the pool is too small.
    """
    pool = [str(question["question"]) for question in questions]
    if sample_mode == "zipf":
        return zipf_queries(pool, steps, seed=seed)
    if sample_mode == "no-repeat":
        if steps > len(pool):
            raise ValueError(
                f"no-repeat evaluation requested {steps} steps but the query pool has "
                f"only {len(pool)} queries")
        indices = random.Random(seed).sample(range(len(pool)), steps)
        return [pool[index] for index in indices]
    raise ValueError(f"unknown query sample mode: {sample_mode}")


def validate_run_scope(purpose: str, real_model: bool, limit: Optional[int], steps: int,
                       targets: int, questions: int, sample_mode: str) -> List[str]:
    """Reject labels that would turn a smoke run into a research claim."""
    errors: List[str] = []
    if purpose not in RUN_PURPOSES:
        errors.append(f"unknown run purpose {purpose!r}")
        return errors
    if real_model and purpose == "smoke":
        errors.append("a real-model run must declare --run-purpose pilot or research")
    if purpose == "pilot":
        if limit != 5000:
            errors.append("integration pilot requires --limit 5000 (or smaller)")
        if questions != 20:
            errors.append(f"integration pilot requires the prepared 20-query pool (got {questions})")
        if steps > 20:
            errors.append("integration pilot supports at most 20 stream steps")
        if targets > 5:
            errors.append("integration pilot supports at most 5 poisoned targets")
    if purpose in {"research", "research_lifecycle"}:
        if limit is not None:
            errors.append("research runs require the full prepared corpus; remove --limit")
        if questions != 500:
            errors.append(f"research runs require the prepared 500-query pool (got {questions})")
        if steps != 500:
            errors.append(f"research runs require exactly 500 stream steps (got {steps})")
        expected = "no-repeat" if purpose == "research" else "zipf"
        if sample_mode != expected:
            errors.append(
                f"{purpose} requires --sample-mode {expected}; the Zipf lifecycle stream "
                "is never the same experiment as a unique-query evaluation")
    return errors


def _dataset_signature(corpus: Path, queries: Path, qrels: Path) -> Dict[str, Any]:
    return {"corpus": _file_identity(corpus), "queries": _file_identity(queries),
            "qrels": _file_identity(qrels)}


def run_signature(config: Config, purpose: str, sample_mode: str, seed: int, steps: int,
                  targets: int, limit: Optional[int], n_sources: int,
                  dataset: Dict[str, Any], target_qids: Sequence[str]) -> str:
    """Stable identity for safe resume: config, data, scope and code."""
    payload = {
        "schema": "ragledger.stream_signature.v1",
        "git_commit": _git_commit(),
        "config": config.model_dump(mode="json"),
        "purpose": purpose,
        "sample_mode": sample_mode,
        "seed": seed,
        "steps": steps,
        "targets": targets,
        "limit": limit,
        "n_sources": n_sources,
        "dataset": dataset,
        "target_qids": list(target_qids),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def check_output_state(out_dir: Path, resume: bool) -> Optional[Dict[str, Any]]:
    """Refuse to reuse any non-empty run directory."""
    if not out_dir.exists():
        return None
    existing = sorted(item.name for item in out_dir.iterdir())
    if not existing:
        return None
    if not resume:
        raise ValueError(
            f"output directory already contains partial artifacts {existing}; "
            "this may include ingestion, index or poison preparation without a "
            "checkpoint. Move/archive the directory and use a new --out; "
            "do not delete and reuse it as if it were a clean run")
    if "metrics.json" in existing:
        raise ValueError("this run already has metrics.json; refusing to overwrite a completed run")
    if "checkpoint.json" not in existing or "run_manifest.json" not in existing:
        raise ValueError("resume requires both checkpoint.json and run_manifest.json; "
                         "partial artifacts have no resumable boundary")
    checkpoint = _read_json(out_dir / "checkpoint.json")
    if checkpoint.get("in_flight_step") is not None:
        raise ValueError(
            "the previous attempt was interrupted inside a stream step; its ledger may "
            "contain partial effects. Restore the latest Drive snapshot or start a new "
            "run directory instead of mixing states")
    return checkpoint


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None,
               timeout: float = 5.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama_provenance(config: Config) -> Dict[str, Any]:
    if config.generation.backend != "ollama":
        return {}
    base, model = config.generation.base_url.rstrip("/"), config.generation.model_name
    r: Dict[str, Any] = {"tag": model, "requested_revision": config.generation.revision}
    def call(path,payload=None): return _http_json(f"{base}{path}", payload)
    try:
        hit = next((x for x in call("/api/tags").get("models", []) if x.get("name") == model), None)
        if hit: r.update({"tag_digest": hit.get("digest"), "tag_modified_at": hit.get("modified_at")})
        else: r["tags_error"] = f"exact tag not found: {model}"
    except Exception as exc: r["tags_error"] = f"{type(exc).__name__}: {exc}"
    try:
        shown = call("/api/show", {"name": model})
        r.update({"details": shown.get("details") or {}, "format": shown.get("format"),
                  "capabilities": shown.get("capabilities") or [], "model_info": shown.get("model_info") or {}})
    except Exception as exc: r["show_error"] = f"{type(exc).__name__}: {exc}"
    try: r["runtime_version"] = call("/api/version").get("version")
    except Exception as exc: r["version_error"] = f"{type(exc).__name__}: {exc}"
    try:
        hit = next((x for x in call("/api/ps").get("models", []) if x.get("name") == model), None)
        if hit:
            r.update({"served_name": hit.get("name"), "served_digest": hit.get("digest"),
                      "size_vram_bytes": int(hit.get("size_vram") or 0),
                      "gpu_resident": int(hit.get("size_vram") or 0) > 0, "expires_at": hit.get("expires_at")})
        else: r["ps_error"] = f"exact running model not found: {model}"
    except Exception as exc: r["ps_error"] = f"{type(exc).__name__}: {exc}"
    r["digest_matches_config"] = bool(r.get("served_digest") and r["served_digest"] == config.generation.revision)
    r["tag_matches_served"] = bool(r.get("tag_digest") and r.get("served_digest") and r["tag_digest"] == r["served_digest"])
    r["version_matches_config"] = bool(not config.generation.serving_version or r.get("runtime_version") == config.generation.serving_version)
    return r


def verify_ollama_runtime(config: Config) -> Dict[str, Any]:
    """Fail closed unless the pulled and currently served digests match the pin."""
    r = _ollama_provenance(config); rev = config.generation.revision or ""; bad = []
    if not rev.startswith("sha256:") or len(rev) < 71: bad.append("revision must be an exact sha256 digest")
    if not r.get("served_digest"): bad.append("cannot read the served digest")
    if not r.get("digest_matches_config"): bad.append("served digest != configured revision")
    if not r.get("tag_matches_served"): bad.append("pulled tag digest != served digest")
    if config.generation.serving_version and not r.get("runtime_version"): bad.append("runtime version unavailable")
    if not r.get("version_matches_config"): bad.append("runtime version != pinned version")
    if bad: raise RuntimeError("Ollama model provenance failed: " + "; ".join(bad))
    return r


def _model_provenance(config: Config, pipeline, verifier) -> Dict[str, Any]:
    embedder = getattr(pipeline, "embedder", None)
    llm = getattr(getattr(pipeline, "generator", None), "llm", None)
    nli_probe = verifier.nli_probe() if getattr(verifier, "mode", None) == "nli" else {}
    ollama = _ollama_provenance(config)
    quantisation = (ollama.get("details", {}).get("quantization_level")
                    or config.generation.quantization or "UNSPECIFIED")
    return {
        "equivalence_claim": False,
        "embedding": {
            "backend": config.embedding.backend,
            "model_id": getattr(embedder, "model_name", config.embedding.model_name),
            "revision": getattr(embedder, "revision", config.embedding.revision) or "UNSPECIFIED",
            "precision": getattr(embedder, "precision", config.embedding.precision),
            "device": getattr(embedder, "device", config.embedding.device),
            "dim": getattr(embedder, "dim", None),
        },
        "generation": {
            "backend": config.generation.backend,
            "model_id": ollama.get("served_name") or getattr(llm, "name", config.generation.model_name),
            "revision": config.generation.revision or "UNSPECIFIED",
            "immutable_revision": bool((config.generation.revision or "").startswith("sha256:") or (config.generation.backend != "ollama" and len(config.generation.revision or "") >= 40)),
            "digest_matches_config": ollama.get("digest_matches_config"),
            "serving_version": config.generation.serving_version,
            "runtime_version": ollama.get("runtime_version"),
            "version_matches_config": ollama.get("version_matches_config"),
            "precision": config.generation.precision or "UNSPECIFIED",
            "quantization": quantisation,
            "device": ("GPU-resident" if ollama.get("gpu_resident") else
                       ("server-managed; verify nvidia-smi" if config.generation.backend == "openai"
                        else getattr(llm, "device", "UNSPECIFIED"))),
            "runtime": ollama,
        },
        "verifier": {
            "backend": getattr(verifier, "mode", None),
            "model_id": nli_probe.get("model", config.security.nli_model),
            "revision": nli_probe.get("revision", config.security.nli_revision)
                        or "UNSPECIFIED",
        },
    }


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _file_identity(path: Path) -> Dict[str, Any]:
    """Identify a dataset file without loading it into memory."""
    identity: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return identity
    identity["size_bytes"] = path.stat().st_size
    # A full SHA-256 is useful provenance, but avoid turning a multi-GB full
    # corpus into an unexpected minutes-long pre-run step.
    if identity["size_bytes"] <= 2 * 1024**3:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        identity["sha256"] = digest.hexdigest()
    else:
        identity["sha256"] = None
        identity["sha256_note"] = ">2 GiB; size and mtime recorded instead"
    identity["mtime_ns"] = path.stat().st_mtime_ns
    return identity


def _hardware() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # type: ignore
        data["torch"] = getattr(torch, "__version__", None)
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            data["gpus"] = [{"name": torch.cuda.get_device_properties(i).name,
                             "total_vram_bytes": int(torch.cuda.get_device_properties(i).total_memory)}
                            for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 - optional provenance
        data["torch"] = None
        data["cuda_available"] = False
    return data


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
    parser.add_argument("--backup-dir", default=None,
                        help="Drive backup root; writes atomic snapshots at prepared/step boundaries")
    parser.add_argument("--restore", action="store_true",
                        help="restore the newest Drive snapshot matching this exact run signature "
                             "before checking or resuming the active run (requires --backup-dir)")
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
                        help="deprecated alias: reuse a prepared run (requires --resume)")
    parser.add_argument("--run-purpose", choices=RUN_PURPOSES, default="smoke",
                        help="explicit scope; real-model runs must be pilot or research")
    parser.add_argument("--sample-mode", choices=("zipf", "no-repeat"), default="zipf",
                        help="zipf samples with replacement (trust lifecycle); no-repeat "
                             "evaluates each sampled query index exactly once")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from checkpoint.json after verifying "
                             "the full config/dataset/model signature")
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
    parser.add_argument("--require-protected", action="store_true",
                        help="fail closed unless the functional doctor is PROTECTED with "
                             "NLI and a persistent ledger (automatic for real backends)")
    args = parser.parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()

    out_dir = Path(args.out)
    if args.skip_index and not args.resume:
        print("--skip-index is no longer sufficient; use --resume for a prepared run",
              file=sys.stderr)
        return 2
    if args.restore and not args.backup_dir:
        print("--restore requires --backup-dir", file=sys.stderr)
        return 2

    config = apply_overrides(Config.load(args.config, storage={"root": str(out_dir)}), args.set)
    # Configure the security stack before building the pipeline.  Building it
    # first and then replacing its ledger opened a second SQLite connection to
    # the same trust database, which deadlocked the first query with
    # "database is locked".
    config.trust.quarantine_refutations = args.quarantine_refutations
    config.trust.hierarchical = not args.no_hierarchical
    config.security.max_corroboration = args.max_corroboration
    real_model = (config.generation.backend != "stub"
                  or config.embedding.backend != "hashing")
    ollama_runtime: Dict[str, Any] = {}
    if config.generation.backend == "ollama":
        try:
            ollama_runtime = verify_ollama_runtime(config)
        except RuntimeError as exc:
            print(f"refusing unverifiable pilot model: {exc}", file=sys.stderr)
            return 2
    sqlite_paths = [config.path(config.storage.provenance_db),
                    config.path(config.storage.answer_db)]
    trust_path = config.trust_db_path()
    if trust_path != ":memory:": sqlite_paths.append(Path(trust_path))
    index_base = str(config.path(config.storage.index_path))
    questions = load_questions(Path(args.queries))
    if not questions:
        print("no questions found", file=sys.stderr)
        return 2
    targets = questions[: max(1, args.targets)]
    qrels_path = Path(args.qrels) if args.qrels else Path(args.queries).with_name("qrels.tsv")
    if args.run_purpose in {"pilot", "research", "research_lifecycle"} and not qrels_path.exists():
        print("pilot/research runs require --qrels; target gold coverage cannot be verified",
              file=sys.stderr)
        return 2
    scope_errors = validate_run_scope(
        args.run_purpose, real_model, args.limit, args.steps, args.targets,
        len(questions), args.sample_mode)
    if scope_errors:
        print("run scope is not valid:", file=sys.stderr)
        for error in scope_errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    dataset_contract: Dict[str, Any] = {}
    expected = {"pilot": (5000, 20), "research": (200000, 500),
                "research_lifecycle": (200000, 500)}
    if args.run_purpose in expected:
        passages, queries = expected[args.run_purpose]
        dataset_contract = verify_subset(Path(args.corpus), Path(args.queries), qrels_path,
                                         passages, queries)
        if not dataset_contract.get("ok"):
            print("dataset contract failed; refusing to start:", file=sys.stderr)
            for error in dataset_contract.get("errors", []): print(f"  - {error}", file=sys.stderr)
            return 2
        print(f"dataset verified: {dataset_contract['corpus_rows']:,} passages, "
              f"{dataset_contract['query_rows']} queries, gold "
              f"{dataset_contract['gold_passages_present']}/{dataset_contract['gold_passages_required']}")
    dataset_identity = _dataset_signature(Path(args.corpus), Path(args.queries), qrels_path)
    dataset_identity["subset_contract"] = dataset_contract
    signature = run_signature(config, args.run_purpose, args.sample_mode, args.seed,
                              args.steps, args.targets, args.limit, args.n_sources,
                              dataset_identity, [target["qid"] for target in targets])
    # Restore is deliberately owned by the runner: the canonical signature has
    # just been computed from the effective configuration, dataset contract and
    # code revision.  A snapshot for any other experiment is never installed in
    # the active directory, and the active directory is checked only afterwards.
    restored = None
    if args.restore:
        restored = restore_latest_snapshot(out_dir, Path(args.backup_dir),
                                           expected_signature=signature)
        if restored is None:
            print(f"no Drive snapshot matches run signature {signature[:12]}…; "
                  "continuing without restoring")
        else:
            print(f"restored matching Drive snapshot for {signature[:12]}… at step "
                  f"{restored.get('completed_steps', 0)}")
            if (out_dir / "metrics.json").exists():
                print("restored a completed run; no steps will be executed")
                return 0
    resume = bool(args.resume or restored is not None)
    try:
        previous_checkpoint = check_output_state(out_dir, resume)
    except ValueError as exc:
        print(f"refusing to mix run outputs: {exc}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    if previous_checkpoint is not None:
        previous_manifest = _read_json(out_dir / "run_manifest.json")
        if previous_checkpoint.get("signature") != signature or \
                previous_manifest.get("signature") != signature:
            print("refusing to resume: config, dataset, model revision, seed, scope or code "
                  "changed since the checkpoint", file=sys.stderr)
            return 2
        print(f"resuming prepared run {out_dir} at step "
              f"{previous_checkpoint.get('completed_steps', 0)}/{args.steps}")

    pipeline = PersonAPipeline.from_config(config, load_existing_index=resume)
    # A real-model run must never proceed on a degraded security stack.  This
    # check runs before ingestion/indexing so a misconfigured GPU run fails in
    # seconds instead of after hours of embedding work.
    try:
        health = pipeline.health_check()
    except Exception as exc:  # noqa: BLE001 - fail closed, not open
        print(f"security health check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        pipeline.close()
        return 2
    if real_model or args.require_protected:
        blockers: List[str] = []
        if health.get("status") != "PROTECTED":
            blockers.append(f"security status is {health.get('status')}, not PROTECTED "
                            f"({health.get('status_reason')})")
        if health.get("verifier_mode") != "nli":
            blockers.append(f"verifier mode is {health.get('verifier_mode')}, not nli")
        if not health.get("trust_persistent"):
            blockers.append("trust ledger is not persistent")
        if blockers:
            print("refusing to run a real-model experiment on an unverified stack:",
                  file=sys.stderr)
            for blocker in blockers:
                print(f"  - {blocker}", file=sys.stderr)
            for failure in health.get("readiness_failures", []):
                print(f"  - readiness: {failure}", file=sys.stderr)
            pipeline.close()
            return 2
    ingestor = Ingestor(pipeline.store, config.ingestion)
    if previous_checkpoint is None:
        # ------------------------------------------------------------ ingest
        banner("1. Ingest the clean corpus with simulated contributors")
        report = ingestor.ingest_beir(args.corpus, SourceAssigner(n_sources=args.n_sources,
                                                                 seed=args.seed),
                                      limit=args.limit, seconds_per_doc=60.0)
        print(f"  ingested: {report.to_dict()}")
        print(f"  store: {pipeline.store.counts()}")

        # ------------------------------------------------------------ index
        banner("2. Index the clean corpus (the GPU step)")
        indexed = pipeline.index_chunks()
        pipeline.save_index()
        print(f"  indexed: {indexed} passages   index size: {len(pipeline.index)}")
    else:
        banner("1-2. Reuse the prepared corpus and index")
        print(f"  store: {pipeline.store.counts()}   index size: {len(pipeline.index)}")
        if len(pipeline.index) == 0:
            print("resume checkpoint exists but the saved index is empty", file=sys.stderr)
            pipeline.close()
            return 2

    if previous_checkpoint is None:
        # ------------------------------------------------------------ poison
        banner("3. Craft poison from each target's gold passage (Person C's helpers)")
        gold_ids = load_qrels(qrels_path) if qrels_path.exists() else {}
        if gold_ids:
            print(f"  qrels: {qrels_path} ({len(gold_ids)} queries with a gold passage)")
        else:
            print("  no qrels file: targets fall back to the retrieved pool")
        wanted = {doc_id for target in targets for doc_id in gold_ids.get(target["qid"], [])}
        gold_texts = fetch_passages(Path(args.corpus), wanted) if wanted else {}
        poison_docs = []
        poison_chunk_ids: set = set()
        poison_info: List[Dict[str, Any]] = []
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
            pipeline.close()
            return 2
        pipeline.index_chunks()
        pipeline.save_index()
        print(f"  index size after poison: {len(pipeline.index)}")
        checkpoint = {
            "schema": "ragledger.checkpoint.v1",
            "signature": signature,
            "phase": "prepared",
            "completed_steps": 0,
            "in_flight_step": None,
            "run_purpose": args.run_purpose,
            "sample_mode": args.sample_mode,
            "steps": args.steps,
            "poison_info": poison_info,
            "poison_chunk_ids": sorted(poison_chunk_ids),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write(out_dir / "checkpoint.json", json.dumps(checkpoint, indent=2))
        preliminary = {
            "schema": "ragledger.run_manifest.v2",
            "state": "running",
            "signature": signature,
            "run_purpose": args.run_purpose,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "steps_requested": args.steps,
            "query_pool_size": len(questions),
            "target_queries": len(targets),
            "config_path": str(args.config),
            "config": config.model_dump(mode="json"),
            "dataset": dataset_identity,
            "models": _model_provenance(config, pipeline, pipeline.verifier),
            "started_at": started_at,
        }
        _atomic_write(out_dir / "run_manifest.json",
                      json.dumps(preliminary, indent=2, default=str))
        if args.backup_dir:
            try:
                backup_snapshot(out_dir, Path(args.backup_dir), 0, sqlite_paths,
                                index_base, copy_index=True)
            except Exception as exc:
                print(f"prepared snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                pipeline.close()
                return 2
    else:
        banner("3. Reuse the checkpointed poison documents")
        poison_info = list(previous_checkpoint.get("poison_info") or [])
        poison_chunk_ids = set(previous_checkpoint.get("poison_chunk_ids") or [])
        poison_docs = poison_info
        print(f"  poison documents: {len(poison_docs)}")

    # ---------------------------------------------------------------- trust
    banner("4. Wire in Person B's trust layer")
    # Use the stack PersonAPipeline.from_config already attached.  It is the
    # same components the CLI and the chronological runner use; creating a
    # second ledger here was the duplicate-connection bug.
    ledger = pipeline.trust_provider
    verifier = pipeline.verifier
    policy = pipeline.policy
    queue = pipeline.security.queue if pipeline.security is not None else None
    if not isinstance(ledger, TrustLedger) or verifier is None or not isinstance(policy, TrustPolicy):
        print("security stack did not attach; refusing to run an unprotected stream",
              file=sys.stderr)
        pipeline.close()
        return 2
    trust_config = ledger.config
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
    stream = build_query_stream(questions, args.steps, args.seed, args.sample_mode)
    semantic = ("Zipf sampling with replacement (trust lifecycle; steps != distinct queries)"
                if args.sample_mode == "zipf" else
                "sampling without replacement (each sampled query index appears once)")
    banner(f"5. Replay the query stream ({len(stream)} steps; {semantic})")
    target_text = {q["question"].strip().lower() for q in targets}

    per_step: List[Dict[str, object]] = []
    stream_path = out_dir / "stream.jsonl"
    if previous_checkpoint is not None:
        checkpoint = previous_checkpoint
        if stream_path.exists():
            per_step = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        if int(checkpoint.get("completed_steps", 0)) != len(per_step):
            print("checkpoint and stream.jsonl disagree; refusing to resume", file=sys.stderr)
            pipeline.close()
            return 2
        if len(per_step) > len(stream):
            print("stream.jsonl has more steps than the configured plan", file=sys.stderr)
            pipeline.close()
            return 2
        print(f"  resuming at step {len(per_step)} of {len(stream)}")
    else:
        checkpoint = _read_json(out_dir / "checkpoint.json")
        _atomic_write(stream_path, "")
    poison_ids = poison_chunk_ids
    for step, question in enumerate(stream):
        if step < len(per_step):
            continue
        checkpoint["in_flight_step"] = step
        checkpoint["phase"] = "streaming"
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(out_dir / "checkpoint.json", json.dumps(checkpoint, indent=2))
        result = pipeline.answer(question, f"stream_{step:04d}")
        drained = policy.drain_verification_queue(verifier)
        cited_poison = [d.doc_id for d in result.retrieval.documents
                        if d.doc_id in poison_ids and d.doc_id in (result.answer or "")]
        context_poison = [doc_id for doc_id in result.decision.context_doc_ids
                          if doc_id in poison_ids]
        retrieved_ids = [d.doc_id for d in result.retrieval.documents]
        row = {
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
            "poison_in_context": context_poison,
            "poison_exposed": bool(context_poison),
            "bands": {band: sum(1 for a in result.assessments if a.band.value == band)
                      for band in ("LOW", "MEDIUM", "HIGH")},
            "verified": [v.outcome.value for v in result.decision.verified],
            "verified_docs": [[v.doc_id, v.outcome.value] for v in result.decision.verified],
            "drained_verified": [v.outcome.value for v in drained],
            "newly_quarantined": list(result.decision.notes.get("newly_quarantined", [])),
            "llm_calls": int(result.llm_calls),
            "latency_ms": float(result.timings_ms.get("total", 0.0)),
        }
        per_step.append(row)
        with open(stream_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint["completed_steps"] = step + 1
        checkpoint["in_flight_step"] = None
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(out_dir / "checkpoint.json", json.dumps(checkpoint, indent=2))
        if args.backup_dir:
            try:
                backup_snapshot(out_dir, Path(args.backup_dir), step + 1,
                                sqlite_paths, index_base)
            except Exception as exc:
                print(f"step snapshot failed at {step + 1}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                pipeline.close()
                return 2

    # ---------------------------------------------------------------- metrics
    banner("6. Smoke-test metrics (not the headline evaluation)")
    target_steps = [row for row in per_step if row["is_target"]]
    poison_answers = [row for row in target_steps if row["poison_cited"]]
    poison_exposed = [row for row in target_steps if row["poison_exposed"]]
    poison_retrieved = [row for row in target_steps if row["poison_retrieved"]]
    eligible_attempts = len(target_steps)

    trust_snaps = ledger.get_trust(sorted(poison_chunk_ids))
    blocked = ledger.blocked_doc_ids()
    poisoned_blocked = sorted(blocked & poison_ids)
    clean_blocked = sorted(blocked - poison_ids)
    first_quarantine = next((row["step"] for row in per_step if row["newly_quarantined"]), None)
    clean_documents = max(0, int(pipeline.store.counts().get("documents", 0))
                          - len(poison_docs))
    true_positive = len(poisoned_blocked)
    false_positive = len(clean_blocked)
    precision = (true_positive / (true_positive + false_positive)
                 if true_positive + false_positive else None)
    recall = true_positive / len(poison_docs) if poison_docs else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    false_positive_rate = false_positive / clean_documents if clean_documents else None

    metrics = {
        "smoke_test": args.run_purpose in {"smoke", "pilot"},
        "run_purpose": args.run_purpose,
        "research_findings_allowed": args.run_purpose == "research",
        "run_kind": "real_model" if real_model else "offline_smoke",
        "git_commit": _git_commit(),
        "seed": args.seed,
        "note": ("component trust-stream run; the chronological runner owns answer-level "
                 "ASR and is NOT MEASURED here (no validated attack-specific "
                 "answer criterion); this runner reports exposure and citation, not "
                 "answer success, and does not implement the full attack x baseline matrix"),
        "bands": "forced MEDIUM (demo mode)" if args.force_medium else "calibrated/heuristic",
        "config": str(args.config),
        "dataset_contract": dataset_contract,
        "corpus": str(args.corpus),
        "steps": len(per_step),
        "query_pool_size": len(questions),
        "stream_steps_requested": args.steps,
        "stream_steps_completed": len(per_step),
        "query_sampling": args.sample_mode,
        "distinct_queries_evaluated": len({row["query"] for row in per_step}),
        "repeated_query_evaluations": (len(per_step)
                                       - len({row["query"] for row in per_step})),
        "target_steps": len(target_steps),
        "verifier_mode": verifier.mode,
        "llm": ("stub (claim extraction is lexical: verdicts are indicative only)"
                if getattr(pipeline.generator.llm, "name", "") == "stub"
                else getattr(pipeline.generator.llm, "name", "unknown")),
        "poison_documents": len(poison_docs),
        "poison_targets": poison_info,
        "eligible_attack_attempts": eligible_attempts,
        "poison_retrieved_attempts": len(poison_retrieved),
        "poison_retrieved_rate": ((len(poison_retrieved) / eligible_attempts)
                                  if eligible_attempts else None),
        "poison_exposed_attempts": len(poison_exposed),
        "poison_exposure_rate": ((len(poison_exposed) / eligible_attempts)
                                 if eligible_attempts else None),
        "poison_cited_attempts": len(poison_answers),
        "poison_citation_rate": ((len(poison_answers) / eligible_attempts)
                                 if eligible_attempts else None),
        **answer_level_asr_fields(),
        "poison_quarantined": len(poisoned_blocked),
        "clean_false_quarantine": len(clean_blocked),
        "clean_false_quarantine_ids": clean_blocked,
        "first_quarantine_step": first_quarantine,
        "document_level": {
            "poison_documents": len(poison_docs),
            "clean_documents": clean_documents,
            "true_positive_quarantine": true_positive,
            "false_positive_quarantine": false_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
            "detection_delay_steps": first_quarantine,
        },
        "mean_llm_calls_per_query": round(statistics.fmean(
            [row["llm_calls"] for row in per_step]) if per_step else 0.0, 3),
        "llm_calls_total": sum(row["llm_calls"] for row in per_step),
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
        "research_evaluation": {
            "implements_full_attack_baseline_matrix": False,
            "remaining_requirements": "docs/PERSON_C_TODO.md",
            "answer_level_asr_status": "NOT MEASURED",
        },
    }
    print(json.dumps(metrics, indent=2))

    _atomic_write(out_dir / "metrics.json", json.dumps(metrics, indent=2))
    manifest = {
        "schema": "ragledger.run_manifest.v2",
        "state": "completed",
        "signature": signature,
        "run_purpose": args.run_purpose,
        "sample_mode": args.sample_mode,
        "research_findings_allowed": args.run_purpose == "research",
        "run_kind": metrics["run_kind"],
        "real_model": real_model,
        "git_commit": metrics["git_commit"],
        "command": sys.argv,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config": config.model_dump(mode="json"),
        "dataset": dataset_identity,
        "dataset_contract": dataset_contract,
        "backup_dir": args.backup_dir,
        "models": _model_provenance(config, pipeline, verifier),
        "seed": args.seed,
        "hardware": _hardware(),
        "verifier_mode": verifier.mode,
        "security": {
            "status": health.get("status"),
            "trust_persistent": health.get("trust_persistent"),
            "trust_db": health.get("trust_db"),
            "readiness_failures": health.get("readiness_failures", []),
        },
        "ledger_path": getattr(ledger, "path", None),
        "indexed_passages": len(pipeline.index),
        "query_pool_size": len(questions),
        "stream_steps_requested": args.steps,
        "stream_steps_completed": len(per_step),
        "query_sampling": args.sample_mode,
        "distinct_queries_evaluated": len({row["query"] for row in per_step}),
        "repeated_query_evaluations": (len(per_step)
                                       - len({row["query"] for row in per_step})),
        "evaluated_queries": len(per_step),
        "target_queries": len(targets),
        "poison_documents": len(poison_docs),
        "resumed": bool(args.resume),
        "resumed_from_step": (int(previous_checkpoint.get("completed_steps", 0))
                              if previous_checkpoint is not None else 0),
        "answer_level_asr_status": "NOT MEASURED",
        "research_evaluation": {
            "full_attack_baseline_matrix": False,
            "remaining_requirements": "docs/PERSON_C_TODO.md",
            "note": "this trust-stream runner is not the complete C6 benchmark",
        },
    }
    _atomic_write(out_dir / "run_manifest.json", json.dumps(manifest, indent=2, default=str))

    # ---------------------------------------------------------------- figures
    banner("7. Trust-dynamics figures (B6)")
    history = ledger.export_history()
    write_history_csv(history, out_dir / "trust_history.csv")
    figures: List[str] = []
    timeline = None
    try:
        figures = plot_trust_dynamics(history, out_dir / "figures")
        timeline = plot_quarantine_timeline(ledger.get_audit_log(limit=10000),
                                            out_dir / "figures" / "quarantine_timeline.png")
    except RuntimeError as exc:
        # matplotlib is an optional reporting dependency; the measured run is
        # already written and must not be lost because a figure cannot render.
        print(f"  figures skipped: {exc}")
    print(f"  history rows: {len(history)}   figures: {len(figures)}")
    for path in figures:
        print(f"    {path}")
    if timeline:
        print(f"    {timeline}")

    if args.backup_dir:
        try:
            backup_snapshot(out_dir, Path(args.backup_dir), len(per_step),
                            sqlite_paths, index_base)
        except Exception as exc:
            print(f"final snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            ledger.close(); pipeline.close()
            return 2

    ledger.close()
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
