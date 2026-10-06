# TRACE-RAG — research prototype

TRACE-RAG combines provenance-aware retrieval, poisoning signals, a trust ledger,
corroboration verification, quarantine/remediation, attack helpers, baseline
policies, evaluation utilities, and cited/abstaining generation for RAG systems.

It runs offline with a hashing embedder and stub answer model; Hugging Face
embeddings, NLI verification, local/remote generation, NumPy/FAISS indexes, and
CPU/CUDA/PyTorch-XLA device selection are configurable. **This is a research
prototype, not production-ready for company deployment.** See
[`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) for the assessment
and remaining security, privacy, reliability, and benchmark work.

---

## First time here?

```bash
cd trace-rag                  # the extracted folder, containing pyproject.toml
python scripts/check_setup.py # says what is installed and what to fix
```

## Running a development subset on a GPU

This existing GPU walkthrough uses a gold-preserving 200k-passage development
subset for faster iteration; it is not a full-NQ evaluation.

```bash
conda env create -f environment.yml && conda activate trace-rag
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -e ".[all,data]"
python scripts/download_data.py --dataset nq --out data/nq --subset 200000
trace-rag --config config/nq_gpu.yaml ingest-beir --corpus data/nq/corpus_subset.parquet
trace-rag --config config/nq_gpu.yaml index
trace-rag --config config/nq_gpu.yaml query "who designed the eiffel tower?"
```

Full CUDA walkthrough: [`docs/RUNNING_ON_GPU.md`](docs/RUNNING_ON_GPU.md).

## Running in Google Colab

- TPU / PyTorch-XLA: [`notebooks/trace_rag_colab_tpu.ipynb`](notebooks/trace_rag_colab_tpu.ipynb) and [`docs/COLAB_TPU_RUNBOOK.md`](docs/COLAB_TPU_RUNBOOK.md).
- CUDA GPU: [`notebooks/trace_rag_colab.ipynb`](notebooks/trace_rag_colab.ipynb) and [`docs/COLAB_RUNBOOK.md`](docs/COLAB_RUNBOOK.md).

The TPU notebook now targets the full 2.68M-passage BEIR NQ corpus and a real
Ollama model, but remains experimental until it completes on live hardware. It
samples query traffic and uses controlled poison mutations; it is an integration
smoke, not a complete attack/baseline/seed benchmark or real-incident study.

## Quick start

```bash
pip install -e ".[dev,faiss]"        # core + tests + FAISS
pytest -q                            # optional model/FAISS tests skip if extras are absent
python scripts/demo_end_to_end.py    # full pipeline on the bundled mini corpus
python scripts/profile_and_ablate.py # A7: latency percentiles + signal ablation
```

The demo ingests a clean corpus, injects PoisonedRAG-style passages from a new
contributor, answers the targeted question before and after training the
suspicion scorer, then quarantines the injected passages and shows retroactive
remediation of the answers already served.

### Command line

```bash
trace-rag ingest       --path data/docs              # PDF / TXT / MD / HTML
trace-rag ingest-beir  --corpus data/nq/corpus-00000-of-00001.parquet --limit 50000
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
| A7 Profiling & ablation | `scripts/profile_and_ablate.py` | Per-stage latency percentiles, LLM calls per query, and held-out AUC with each signal switched off. |
| L7 Grounded generation | `generation/` | Mandatory `[passage-id]` citations, hallucinated citations rejected, abstention when the trust-weighted evidence mass is too low. |

The verifier and policy layers are implemented under `src/trace_rag/trust/`;
null/default interfaces remain available for offline tests and ablations. Attack,
baseline, stream, and evaluation helpers are under `src/trace_rag/attacks/`,
`baselines/`, and `evaluation/`.

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
trace-rag --config config/full_nq.yaml ingest-beir --corpus data/nq/corpus-00000-of-00001.parquet
trace-rag --config config/full_nq.yaml index
```

A flat FAISS index over NQ (2.68M × 768 float32) needs about 8.2 GB of RAM; the
IVF-PQ settings above bring that down by roughly an order of magnitude. Develop
on `--limit 500000` first.

---

## Tests

```bash
pytest -q
pytest -q --cov=trace_rag --cov-report=term-missing   # inspect current coverage locally
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
  trust/              # ledger, corroboration verifier, policy, queue
  attacks/            # poison construction and adaptive streams
  baselines/          # baseline policies
  evaluation/         # stream adapters, metrics, and statistics
  pipeline.py         # PersonAPipeline: the assembled system
  cli.py              # trace-rag command line
config/  docs/  examples/  scripts/  tests/
```

## Honest limits

* A historical Natural Questions subset was run through ingestion, Contriever,
  FAISS, and retrieval on a laptop GPU; it exposed earlier scale defects listed
  in `docs/VERIFICATION.md`. This does not establish full-NQ performance or the
  current TPU path.
* The hashing embedder and stub LLM are for offline tests and deterministic
  plumbing checks. They are not valid evidence for answer quality or benchmark
  accuracy.
* The bundled mini corpus is trivially separable. Its smoke metrics are
  regression evidence only; the generic evaluation matrix is not yet a complete,
  validated attack × baseline × seed benchmark.
* `docs/VERIFICATION.md` lists code-level defects and test evidence;
  `docs/REQUIREMENTS.md` and `docs/PERSON_C_TODO.md` record research gaps.
* Signals S1 and S3 are based on published observations (PoisonedRAG's
  construction and TrustRAG's clustering). Attribution and benchmark claims
  must be checked against the cited papers and evaluated protocol before use.

## Status

**Research prototype — not production-ready.** The repository has unit and
integration tests and a runnable CPU smoke. The sandbox test suite passes, with
optional FAISS/model/plot tests skipped when their extras are unavailable. No
live TPU/XLA run has been performed in this sandbox; the Colab TPU notebook is
experimental and must prove actual XLA device use in the user's runtime.

Before presenting research conclusions, run the controlled benchmark on the
selected data with real model revisions, complete baseline parity, independent
seeds, and confidence intervals. Before company deployment, complete the
security, privacy, multi-tenant, scale, and reliability work in
[`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md).
