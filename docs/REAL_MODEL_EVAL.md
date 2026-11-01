# Real-model Natural Questions run

This is the protocol for RAGLedger's first genuine model-backed experiment.
**It has not been executed.** Everything below is a runbook and preflight
record, not a report of real-model results.

> **Status: NOT EXECUTED.**
>
> * Starting commit for this pass: `9bf18ca69f423ae2d5770c3f472b3d7357ef1638`.
> * Earlier preflight/subset commits: `e84f60390f7011bf69eee1a3d42ff6a54b6a1901` and
>   `58de432c3124cc7cf81dcbfc4cc169b91c00af08`.
> * Current Colab workflow/pinned implementation commit: `32333fe125c446d043bd422e9bbd86005d1259d8`.
> * Research scope: a **200,000-passage / 500-query subset** of Natural
>   Questions, not the full 2.68M-passage corpus.
> * Current preflight: **BLOCKED** on this Windows host.
> * No real embedding, generation or NLI result exists in this repository.
> * Answer-level ASR is **NOT MEASURED**; exposure and citation are not answer
>   success and must never be labelled ASR.

The deterministic fixture matrix and the mini trust stream are separate,
clearly labelled smoke scopes. Their numbers are not NQ evidence.

---

## 1. Two scopes, two names

| | Integration pilot | Research experiment |
|---|---|---|
| Purpose | prove the complete real-model path works | measure poisoning behaviour on NQ |
| Data | 5,000 passages, 20 queries | 200,000 passages, 500 queries |
| Config | `config/pilot_gpu.yaml` (or explicitly alternate `config/colab_pilot_qwen.yaml`) | `config/nq_gpu.yaml` |
| Output | `runs/pilot/trust_stream/` (or a separate Colab pilot directory) | `runs/nq/trust_stream/` |
| Claim status | engineering validation only | research result, only after review |

Do not merge the two result directories and do not describe pilot metrics as
research findings.

The local/GPU pilot keeps the research model. The Colab integration pilot is an
**explicitly named alternate-model pilot**: Contriever + DeBERTa NLI + Qwen2.5-7B
via Ollama q4_K_M. Qwen and Llama are not equivalent; their manifests and result
directories must remain separate. The research thresholds and policy are unchanged.

For the full Colab workflow, pinned commits, Drive persistence, model revisions
and resume rules, see [`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md).

---

## 2. What the run measures

The trust-stream runner reports:

* poison retrieval, context exposure and answer citation, with explicit
  eligible-attempt numerators and denominators;
* document-level quarantine precision, recall, F1, false-positive rate and
  first-quarantine step;
* SUPPORT / REFUTE / NEUTRAL verification counts;
* trust dynamics and persistent ledger state;
* LLM calls, latency and the actual verifier/model mode.

It does **not** score the generated answer against the attacker's intended
answer. Therefore `answer_level_asr` remains `null` with status
`NOT MEASURED`. A real ASR matrix still requires a validated,
attack-specific answer criterion and an attack × baseline runner.

---

## 3. Prerequisites and preflight

Required: Python 3.10+, a CUDA GPU with enough VRAM for Contriever, DeBERTa and
Llama-3.1-8B, roughly 20+ GiB free disk (more for the full source corpus), and
`torch`, `transformers`, `faiss`, `pyarrow` and `huggingface_hub`.

```bash
pip install -e ".[models,data]"
```

Run the non-invasive preflight first. It never downloads weights and never
starts a service:

```bash
python scripts/check_research_environment.py --config config/pilot_gpu.yaml
python scripts/check_research_environment.py --config config/nq_gpu.yaml
```

Exit codes: `0 READY`, `1 DEGRADED`, `2 BLOCKED`. Static checks alone can only
reach `DEGRADED`, because a model name in YAML is not proof that inference
works. To allow `READY`, run the functional doctor probes with HF offline flags:

```bash
python scripts/check_research_environment.py --config config/nq_gpu.yaml \
  --functional --preflight-storage-root /tmp/ragledger-preflight \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv
```

Current host output: **BLOCKED**. Blocking checks: missing `torch`,
`transformers` and `faiss`; no CUDA; `facebook/contriever` absent from
the local cache; no generator served at `localhost:8000`. `pyarrow 26.0.0` is
installed locally but the Colab target pin `17.0.0` is still unverified. The NLI
model file is cached, but `transformers` is absent, so NLI inference cannot run.
This is an environment verdict, not a model evaluation.

### Endpoint and model rules

* vLLM: `generation.backend: openai`, `base_url: http://localhost:8000/v1`,
  `model_name: meta-llama/Llama-3.1-8B-Instruct`.
* Ollama: `generation.backend: ollama`, `base_url: http://localhost:11434`,
  and an Ollama model name such as `llama3.1:8b` via an explicit `--set`.
* The endpoint check lists served models; the functional doctor additionally
  performs a two-token completion.
* The NLI model and revision are configuration fields (`security.nli_model`,
  `security.nli_revision`) and are recorded in the manifest. If NLI cannot load,
  the verifier falls back to lexical, the doctor reports `DEGRADED`, and the
  real-model gate refuses the run.

---

## 4. Prepare the data

```bash
python scripts/download_data.py --dataset nq --out data/nq \
  --subset 200000 --queries 500
```

This command first downloads the BEIR NQ source corpus, queries and qrels, then
builds a subset that keeps every gold passage for the 500 sampled queries. It is
resumable; `--skip-download` rebuilds the subset from files already on disk.
The source corpus is large: do not run this step on a constrained disk, and do
not commit `data/`.

Expected files:

```
data/nq/corpus.parquet
data/nq/queries.parquet
data/nq/qrels.tsv
data/nq/corpus_subset.parquet
data/nq/queries_subset.jsonl
```

The runner accepts `queries_subset.jsonl` (fields `_id`/`text`) and
`qrels.tsv`; both are produced by the downloader. The 200K subset is a subset,
not the full corpus, and must be described that way.

For the pilot, use `--out data/nq_pilot --subset 5000 --queries 20`.

---

## 5. Start the generator

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

The functional preflight must report `PROTECTED` with `verifier_mode: nli` and
`trust_persistent: true`. The trust-stream runner enforces this automatically
for any non-stub generation backend, before ingestion:

```bash
python scripts/run_trust_stream.py --config config/pilot_gpu.yaml ...
# exits 2 with an explicit blocker list if the stack is DEGRADED/UNPROTECTED
```

`--require-protected` applies the same gate explicitly. There is no override
that turns a stub LLM or a fake NLI verifier into a real-model run.

---

## 6. Integration pilot

```bash
python scripts/download_data.py --dataset nq --out data/nq_pilot \
  --subset 5000 --queries 20
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

python scripts/check_research_environment.py --config config/pilot_gpu.yaml \
  --functional --corpus data/nq_pilot/corpus_subset.parquet \
  --queries data/nq_pilot/queries_subset.jsonl --qrels data/nq_pilot/qrels.tsv

python scripts/run_trust_stream.py --config config/pilot_gpu.yaml \
  --corpus data/nq_pilot/corpus_subset.parquet \
  --queries data/nq_pilot/queries_subset.jsonl \
  --qrels data/nq_pilot/qrels.tsv \
  --run-purpose pilot --sample-mode zipf \
  --limit 5000 --steps 12 --targets 2 --out runs/pilot/trust_stream \
  --require-protected
```

This is an engineering integration test. Report it as "pilot, real backends,
not a research result". Its manifest must say `run_kind: real_model`,
`run_purpose: pilot`, and `equivalence_claim: false`.

The runner itself ingests the clean corpus, builds the FAISS index, injects the
two attack passages, probes the NLI verifier and replays the query stream.
There is no need to run separate CLI ingest/index steps first.

---

## 7. Research experiment (200K / 500 unique queries)

```bash
python scripts/download_data.py --dataset nq --out data/nq \
  --subset 200000 --queries 500
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

python scripts/check_research_environment.py --config config/nq_gpu.yaml \
  --functional --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv

python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl \
  --qrels data/nq/qrels.tsv \
  --run-purpose research --sample-mode no-repeat \
  --steps 500 --targets 5 --out runs/nq/trust_stream
```

This evaluates 500 query indices without replacement: 500 stream steps and 500
distinct queries. The manifest records both. The historical 500-step Zipf
protocol is a **separate** `research_lifecycle` run; it samples with replacement
and must never be reported as 500 unique queries.

`--targets 5` means only five target questions are poisoned in this stream.
It is not the seven-attack × eight-baseline matrix. Do not report the stream as
the full NQ benchmark; answer-level ASR remains **NOT MEASURED**.

### Resume

Use `--resume` with the exact original arguments. The runner validates the Git
commit, resolved config, model revisions, dataset hashes, seed, scope and query
plan before continuing from `stream.jsonl`:

```bash
python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv \
  --run-purpose research --sample-mode no-repeat \
  --steps 500 --targets 5 --out runs/nq/trust_stream --resume
```

`checkpoint.json` is written before and after each step. If the previous process
died inside a step (`in_flight_step` non-null), resume fails closed because the
ledger may contain partial effects; start a new run directory. A completed run
with `metrics.json` is never overwritten. `--skip-index` alone is no longer
supported.

---

## 8. Outputs and provenance

```
runs/nq/trust_stream/
  metrics.json          exposure/citation; document-level precision, recall,
                        F1, FPR, detection delay; verification counts; latency
  stream.jsonl          per-query retrieved/context/cited ids and outcomes
  run_manifest.json     v2: commit, command, dataset hashes, resolved config,
                        exact models/revisions/precision/quantisation, purpose,
                        query sampling and counts, hardware, ledger path
  trust_history.csv     persistent-ledger history
  figures/              optional matplotlib plots
```

`run_manifest.json` is written by the runner for both real and offline modes.
The offline mini run is explicitly `run_kind: offline_smoke`, `real_model:
false`, with hashing embeddings, a stub LLM and lexical verification.

---

## 9. Acceptance checks for a real run

1. Preflight is `READY` and `doctor` is `PROTECTED` after functional probes.
2. `verifier_mode` is `nli`; a lexical fallback invalidates the NLI claim.
3. `trust_persistent` is true and the ledger path survives restart.
4. `run_manifest.json` records the commit, dataset hashes, config, seed, model
   names/backends, hardware and counts.
5. `metrics.json` reports exposure/citation with explicit denominators, plus
   document precision/recall/F1/FPR and quarantine delay.
6. `answer_level_asr` is `null` / `NOT MEASURED`; no ASR claim is derived from
   exposure or citation.
7. Clean false quarantines are reported, not hidden.
8. The run is repeated across seeds before any claim of improvement.

The fixture statistics module supports bootstrap confidence intervals and
McNemar's exact test, but no real-NQ multi-seed run exists.

---

## 10. Remaining blockers

* No CUDA GPU and no `torch`, `transformers`, `faiss` or `pyarrow` on this host.
* No Contriever weights in the local cache and no vLLM/Ollama endpoint serving
  the configured model.
* No real NQ subset is present; downloading the full source corpus was not
  performed.
* The integration pilot has not run; the 200K/500 experiment has not run.
* No real-NQ attack × baseline matrix, trained scorer or multi-seed statistics.
* Answer-level ASR remains unimplemented and unmeasured by design.

Do not claim research validation until the model-backed experiment runs and its
results are independently checked.
