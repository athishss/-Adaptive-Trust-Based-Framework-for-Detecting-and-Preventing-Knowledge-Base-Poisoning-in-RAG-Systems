# TRACE-RAG — Person A

Provenance-aware retrieval, cheap poisoning signals and grounded generation for
the TRACE-RAG secure-RAG project.

This repository is **Person A's half of the system**: everything from a raw
document to a cited answer, plus the detection layer that decides which passages
deserve expensive verification. Person B (trust ledger, corroboration-gated
verifier, quarantine) and Person C (attacks, baselines, evaluation) plug in
through `trace_rag.contracts` — see [`docs/INTERFACES.md`](docs/INTERFACES.md).

It runs end to end with no model downloads (hashing embedder + offline stub LLM),
so tests and the demo work anywhere; swap two lines of config for Contriever +
FAISS + Llama-3.1-8B when you run real experiments.

---

## Quick start

```bash
pip install -e ".[dev,faiss]"        # core + tests + FAISS
pytest -q                            # 130 tests
python scripts/demo_end_to_end.py    # full pipeline on the bundled mini corpus
```

The demo ingests a clean corpus, injects PoisonedRAG-style passages from a new
contributor, answers the targeted question before and after training the
suspicion scorer, then quarantines the injected passages and shows retroactive
remediation of the answers already served.

### Command line

```bash
trace-rag ingest       --path data/docs              # PDF / TXT / MD / HTML
trace-rag ingest-beir  --corpus data/nq/corpus.jsonl --limit 50000
trace-rag index
trace-rag query        "who designed the eiffel tower?"
trace-rag train-scorer --rows runs/nq/labelled_rows.jsonl
trace-rag remediate    --doc-ids poison_1#0000
trace-rag stats
```

### Library

```python
from trace_rag import Config, PersonAPipeline

pipeline = PersonAPipeline.from_config(Config.load("config/default.yaml"))
result = pipeline.answer("Who designed the Eiffel Tower?")
print(result.answer, result.record.abstained, result.llm_calls)
```

---

## What this half does

| Layer | Module | Summary |
|---|---|---|
| L0 Ingestion & provenance | `ingestion/` | PDF/TXT/MD/HTML and BEIR JSONL → ~100-word chunks carrying `doc_id`, `source_id`, `family_id`, SHA-256, version, ingestion time. Near-duplicate *families* come from a self-contained MinHash + LSH. |
| L1 Retrieval | `retrieval/`, `index/`, `embeddings/` | Candidate pool of 50, re-ranked by `similarity × t_eff^λ` down to top-5. Quarantined passages are removed. A trust floor keeps new honest sources reachable. |
| L2 Cheap signals | `detection/signals.py` | Six signals, no LLM calls: query echo, similarity outlier, cluster tightness, ingestion burst, source immaturity, neighbourhood density. |
| L3 Suspicion scorer | `detection/scorer.py` | Logistic regression + isotonic calibration, with band thresholds chosen under **two** constraints at once: HIGH-band false-positive rate ≤ target and mean escalations per query ≤ budget. |
| L5 Answer provenance | `provenance/` | Every answer records the passages it used and the trust values at the time; quarantining a passage flags every past answer that relied on it. |
| L7 Grounded generation | `generation/` | Mandatory `[passage-id]` citations, hallucinated citations rejected, abstention when the trust-weighted evidence mass is too low. |

Layers L4 (verifier) and L6 (policy/quarantine) belong to Person B; this repo
ships working null/default implementations so the pipeline runs without them and
so Person B has a reference to match.

### Design decisions worth knowing

* **Trust is read, never written here.** Only Person B's verified outcomes update
  the ledger. Cheap signals decide what to *check*, never what to trust — that is
  what stops the detector reinforcing its own false positives.
* **Evidence mass aggregates by source.** Five agreeing passages from one
  contributor count once (noisy-OR over per-source maxima), so a swarm of
  injected passages cannot manufacture confidence.
* **S6 uses the isolation tail only on large corpora** (`min_corpus_for_isolation`),
  because on a small corpus every genuinely unique fact looks isolated and would
  be punished.
* **Determinism.** Fixed seeds, sorted tie-breaks, stable chunk ids; the same
  config on two machines produces the same ranking and the same answer.

---

## Switching to real models

`config/full_nq.yaml` is the full-corpus configuration:

```yaml
embedding: { backend: huggingface, model_name: facebook/contriever, device: cuda }
index:     { backend: faiss, faiss_kind: ivfpq, nlist: 4096, pq_m: 96 }
generation:{ backend: openai, model_name: meta-llama/Llama-3.1-8B-Instruct,
             base_url: http://localhost:8000/v1 }
```

```bash
pip install -e ".[all]"
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000     # or: ollama run llama3.1:8b
trace-rag --config config/full_nq.yaml ingest-beir --corpus data/nq/corpus.jsonl
trace-rag --config config/full_nq.yaml index
```

A flat FAISS index over NQ (2.68M × 768 float32) needs about 8.2 GB of RAM; the
IVF-PQ settings above bring that down by roughly an order of magnitude. Develop
on `--limit 500000` first.

---

## Tests

```bash
pytest -q                                    # 130 tests
pytest -q --cov=trace_rag --cov-report=term-missing   # 89% coverage
```

What the suite actually checks, beyond the usual unit tests:

* numpy and FAISS return identical top-k (FAISS is not trusted blindly);
* quarantined passages never reach retrieval, the context, or a citation;
* the verifier is called for MEDIUM-band passages only, and its LLM calls are
  counted into the per-query total;
* threshold selection honours both the FPR cap and the escalation budget;
* splitting by question raises `LeakageError` if a question would appear in two
  splits, and a feature snapshot captured after its query is rejected;
* generation abstains on empty context, invalid citations, unsupported citations
  and low evidence mass — and abstention costs no LLM call;
* remediation flags exactly the affected answers, is idempotent, and survives a
  database reopen.

## Scale and cost

Measured on 2 vCPU with the offline embedder (see [`docs/VERIFICATION.md`](docs/VERIFICATION.md)):

| Operation | Rate | Extrapolated to NQ (2.68M passages) |
|---|---|---|
| Ingestion, `family_backend: exact` | ~13,000 passages/s | ~3.5 min, flat memory |
| Ingestion, `family_backend: minhash` | ~2,200 passages/s | ~20 min, ~5 GB RAM |
| Query without the LLM | 5.7 ms mean, 6.4 ms p95 | signals dominate (~4.3 ms) |

`family_backend` decides how near-duplicate families are found: `minhash` links
paraphrased copies (better, memory-hungry), `exact` links verbatim copies only
and is what the full-corpus config uses. Embedding and LLM time are extra and
must be measured on your GPU.

## Repository layout

```
src/trace_rag/
  contracts.py        # shared dataclasses + protocols  <- the integration surface
  config.py           # typed config (pydantic); every threshold lives here
  ingestion/          # parsers, chunking, provenance store (SQLite)
  embeddings/         # hashing (offline) and HuggingFace (Contriever/BGE)
  index/              # numpy exact index and FAISS (flat / IVF-PQ / HNSW)
  retrieval/          # trust-weighted retriever
  detection/          # six signals, calibrated scorer, leakage-guarded training
  generation/         # prompts, LLM backends, citations, grounded generation
  provenance/         # answer log and retroactive remediation
  pipeline.py         # PersonAPipeline: the assembled system
  cli.py              # trace-rag command line
config/  docs/  examples/  scripts/  tests/
```

## Honest limits

* The hashing embedder and stub LLM exist for tests and the demo. No number from
  them belongs in the report.
* The bundled mini corpus is trivially separable; real evaluation is NQ +
  PoisonedRAG with leave-one-attack-family-out, run by Person C.
* Six defects were found and fixed during a verification pass, including a
  leakage bug in the leave-one-attack-out split and a stale-vector bug in the
  FAISS index; `docs/VERIFICATION.md` lists them and what is still unverified.
* Signals S1 and S3 come from published observations (PoisonedRAG's construction,
  TrustRAG's clustering). The contribution here is their combination with source
  history and the leakage-guarded, constraint-solving calibration — that framing
  is what goes in the paper.
