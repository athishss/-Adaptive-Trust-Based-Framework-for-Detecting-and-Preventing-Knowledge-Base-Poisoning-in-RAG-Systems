# Real-model Natural Questions run

This is the protocol for RAGLedger's first genuine model-backed experiment.
**It has not been executed.** Everything below is a runbook and preflight
record, not a report of real-model results.

> **Status: NOT EXECUTED.**
>
> * Starting commit for this pass: `9bf18ca69f423ae2d5770c3f472b3d7357ef1638`.
> * Code/preflight commit: `e84f60390f7011bf69eee1a3d42ff6a54b6a1901`.
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
| Config | `config/pilot_gpu.yaml` | `config/nq_gpu.yaml` |
| Output | `runs/pilot/trust_stream/` | `runs/nq/trust_stream/` |
| Claim status | engineering validation only | research result, only after review |

Do not merge the two result directories and do not describe pilot metrics as
research findings.

The pilot is deliberately not a smaller copy of the scientific configuration:
it uses the same Contriever, Llama-3.1-8B-Instruct and DeBERTa NLI backends,
but a small subset and a different storage root. The research thresholds and
policy are unchanged.

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
  --functional --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv
```

Current host output: **BLOCKED**. Blocking checks: missing `torch`,
`transformers`, `faiss` and `pyarrow`; no CUDA; `facebook/contriever` absent from
the local cache; no generator served at `localhost:8000`. The NLI model file is
cached, but `transformers` is absent, so NLI inference cannot run. This is an
environment verdict, not a model evaluation.

### Endpoint and model rules

* vLLM: `generation.backend: openai`, `base_url: http://localhost:8000/v1`,
  `model_name: meta-llama/Llama-3.1-8B-Instruct`.
* Ollama: `generation.backend: ollama`, `base_url: http://localhost:11434`,
  and an Ollama model name such as `llama3.1:8b` via an explicit `--set`.
* The endpoint check lists served models; the functional doctor additionally
  performs a two-token completion.
* The NLI loader is hard-coded to `cross-encoder/nli-deberta-v3-base` in
  `trust/verifier.py`. If it cannot load, the verifier falls back to lexical,
  the doctor reports `DEGRADED`, and the real-model gate refuses the run.

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
  --steps 20 --targets 2 --out runs/pilot/trust_stream --require-protected
```

This is an engineering integration test. Report it as "pilot, real backends,
not a research result". Its manifest must say `run_kind: real_model`.

The runner itself ingests the clean corpus, builds the FAISS index, injects the
two attack passages, probes the NLI verifier and replays the query stream.
There is no need to run separate CLI ingest/index steps first.

---

## 7. Research experiment (200K / 500)

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
  --steps 500 --targets 3 --out runs/nq/trust_stream
```

`--targets 3` means only three target questions are poisoned in this stream.
It is not the seven-attack × eight-baseline matrix. Do not report the stream as
the full NQ benchmark.

### Resume

Use `--skip-index` to reuse an existing store and saved index:

```bash
python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv \
  --steps 500 --targets 3 --out runs/nq/trust_stream --skip-index
```

This skips clean re-ingestion and re-indexing, but the script still re-crafts
and re-ingests the poison documents before streaming. Use it only after a
completed run with a valid `provenance.sqlite3` and index in the same output
root. It is not a checkpoint for a partially completed poison stream.

The CLI `index` command supports `--save-every` and `--limit` for building a
large index in sittings; the stream runner expects the index and store to be in
sync.

---

## 8. Outputs and provenance

```
runs/nq/trust_stream/
  metrics.json          exposure/citation; document-level precision, recall,
                        F1, FPR, detection delay; verification counts; latency
  stream.jsonl          per-query retrieved/context/cited ids and outcomes
  run_manifest.json     commit, command, dataset hashes, resolved config, model
                        backends, seed, hardware, ledger path, indexed passages
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
