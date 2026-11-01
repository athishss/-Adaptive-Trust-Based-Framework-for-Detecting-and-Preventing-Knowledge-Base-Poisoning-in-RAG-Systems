# Running RAGLedger on Google Colab (GPU)

The executable notebook is [`notebooks/trace_rag_colab.ipynb`](../notebooks/trace_rag_colab.ipynb).
This runbook documents the same workflow as copy-paste commands and states what
is, and is not, validated. The notebook has **not** been executed in Colab; the
repository validation is the offline test suite (including direct Parquet subset
and snapshot-recovery tests), notebook JSON/AST/pin checks, coverage,
ResourceWarning and compileall checks, and an offline mini stream.

> **Scope:** the Colab pilot uses an explicitly different generator,
> **Qwen2.5-7B via Ollama q4_K_M**, to fit a free T4. It is an integration pilot,
> not the research configuration. The research configuration is Contriever +
> DeBERTa NLI + **Llama-3.1-8B-Instruct**, pinned revision
> `a09a35458c702b33eeacc393d103063234e8bc28`, unquantised bfloat16 via vLLM.
> These two model configurations are not equivalent and their outputs must
> never be pooled.

## 1. Repository and pinned commit

The notebook clones:

```text
https://github.com/athishss/-Adaptive-Trust-Based-Framework-for-Detecting-and-Preventing-Knowledge-Base-Poisoning-in-RAG-Systems.git
```

It pins an explicit full commit SHA in `PINNED_COMMIT`, fetches that object,
checks out detached HEAD, asserts `git rev-parse HEAD` equals the pin, and prints
the actual SHA before any experiment. Do not replace this with a branch name or
`git pull` for a reported run.

The notebook pins implementation commit
`8bf874a10ea2f95b30f8b3e7de15aacc99e7baa0`. That commit contains the exact
implementation and tests described here; the later notebook/documentation
commit only changes the pin and documentation, so it is not pinned to itself.
The notebook was **not** run in Colab.

## 2. Drive persistence

Mount Drive and create `MyDrive/RAGLedger/`:

```text
MyDrive/RAGLedger/
  datasets/raw/          raw NQ BEIR files (downloaded once)
  datasets/pilot/        exact 5,000-passage / 20-query subset
  datasets/research/     exact 200,000-passage / 500-query subset
  configs/               model pins, immutable digest lock, config snapshots
  preflight/             preflight JSON and logs
  backups/<run-id>/      atomic SQLite/checkpoint/index snapshots
  exports/               archives of completed results
  logs/                  vLLM and service logs

Active execution state is temporary. The FAISS index, provenance/answer/trust
SQLite databases, `checkpoint.json` and `stream.jsonl` live in
`/content/ragledger_active/runs/<run-id>/`. They are never executed from the
Drive FUSE mount, and `/content` does **not** survive a Colab runtime reset.
Temporary Hugging Face weights stay in `/content/hf_cache`.

Persistent Drive storage under `MyDrive/RAGLedger/` holds the raw and prepared
datasets, model/config locks, preflight results, completed-step snapshots,
SQLite backups, the prepared index snapshot, exports and logs. At preparation
and after every completed stream step the runner writes an atomic snapshot
under `backups/<run-id>/`: checkpoint, manifest, stream, consistent SQLite
backups and the prepared index. The checkpoint, stream row count and SQLite
snapshot all describe the same completed step. The prepared index is copied
once; later snapshots reference its digest without duplicating hundreds of
megabytes at every step. The active `/content` index is therefore copied **into**
the persistent snapshot; it is not itself a Drive run directory.

After a runtime reset, the notebook invokes the runner with the same arguments
plus `--restore --backup-dir <backup-dir>`. The runner computes the canonical
run signature from the resolved config, model revisions, dataset contract,
code commit, purpose, sampling, seed, scope and source count, then restores only
a snapshot with that exact signature. Snapshot metadata records the run
signature, manifest hash, index base/digest/file set, checkpoint stream position
and SQLite state. A mismatched snapshot is never installed into the active
directory, and an active run with a different signature is never replaced. A
shared backup directory whose prepared index belongs to another signature is
refused; use one backup directory per run.

Replacement uses a staging directory and installs `checkpoint.json` last as the
commit marker; an existing index is kept as a recoverable `.<name>.previous`
directory until the new one is installed. An interrupted backup or restore is
detected and cleaned or refused rather than being silently resumed. The runner
itself refuses any non-empty directory with partial ingestion, index or poison
artifacts, even when no checkpoint or manifest exists. Safe recovery is:
preserve the partial active directory for diagnosis, restore the latest valid
matching snapshot, or create a new run directory. Never delete an incomplete
directory and reuse its name as a clean run, and never hand-edit
`checkpoint.json` to force a resume.

## 3. Install and models

```bash
cd /content/RAGLedger
pip install -q -e ".[all,data]" matplotlib
```

The notebook checks the GPU model, total/free VRAM, CUDA/PyTorch versions, RAM,
disk, and asserts CUDA is available. `faiss-cpu` is installed: **FAISS indexing
is CPU-only** and must not be described as GPU-accelerated. The embedder and NLI
model use CUDA when available; the pilot generator is checked for Ollama GPU
residency (`size_vram > 0`).

Pinned embedding/NLI revisions:

* `facebook/contriever` @ `2bd46a25019aeea091fd42d1f0fd4801675cf699`
* `cross-encoder/nli-deberta-v3-base` @ `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`

The notebook calls `snapshot_download(..., revision=...)` for both **before**
the functional preflight. Preflight then runs with `HF_HUB_OFFLINE=1`, so a
missing or different cache is a blocker, never a silent download.

Pilot generator:

```bash
ollama pull qwen2.5:7b
```

The notebook records the Ollama manifest digest and actual quantisation from
`/api/tags` and `/api/show`, then passes both to preflight and runner as
`generation.revision` and `generation.quantization`.

Research generator:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.55
```

The research weight download is gated behind `RUN_RESEARCH=True` in the
notebook. A free T4 is not a supported host for this model plus Contriever and
DeBERTa.

## 4. Data preparation

The raw BEIR NQ source is downloaded once to `datasets/raw`; subsets are built
with `build_exact_subset` / `write_queries_exact`, always retaining every gold
passage for the selected queries. Construction streams the Parquet corpus with
bounded memory, refuses duplicate or blank document/query IDs, rejects missing
gold passages and insufficient rows, writes the output atomically, and then
`verify_subset` independently checks the exact row count, unique IDs, query
membership and 100% gold coverage.

* Pilot: exactly 5,000 passages, 20-query pool.
* Research: exactly 200,000 passages, 500-query pool.

Downloading the raw corpus is deliberately **opt-in**:
`RUN_DATASET_DOWNLOAD = False` in the notebook. If any of `corpus.parquet`,
`queries.parquet` or `qrels.tsv` is missing, the notebook raises an explanatory
error instead of silently consuming several gigabytes of Drive space. Set the
flag to `True` after checking free space, or copy existing raw files into
`datasets/raw`. `RUN_RESEARCH` remains a separate opt-in.

## 5. Functional preflight

Both preflight and runner receive the **same config and the same `--set`
overrides**. The preflight writes a resolved temporary YAML and the doctor reads
that exact file, eliminating the check-one-config/run-another failure.

The functional doctor constructs a real pipeline and therefore creates SQLite
files. It must not write them into the experiment directory, because the runner
refuses a non-empty output directory without a checkpoint. Pass a disposable
parent directory:

```bash
python scripts/check_research_environment.py \
  --config config/colab_pilot_qwen.yaml --functional --json \
  --preflight-storage-root "$PREFLIGHT_PARENT/pilot" \
  --corpus "$PILOT/corpus_subset.parquet" \
  --queries "$PILOT/queries_subset.jsonl" --qrels "$PILOT/qrels.tsv" \
  --set embedding.revision=2bd46a25019aeea091fd42d1f0fd4801675cf699 \
  --set security.nli_revision=6c749ce3425cd33b46d187e45b92bbf96ee12ec7 \
  --set generation.revision="$QWEN_DIGEST" \
  --set generation.quantization="$QWEN_QUANT" \
  --set storage.root="$PILOT_RUN"
```

The doctor copies the resolved config with model, security and generation
settings unchanged, points every storage path at a unique child of
`--preflight-storage-root`, rebases absolute or `..` storage paths to safe
basenames, and records the result under
`functional_doctor.storage_isolation` in the JSON (`schema`,
`intended_root`, `probe_root`, `rebased_paths`, `cleaned`). The child is
deleted afterwards. The notebook asserts that a fresh experiment directory is
still absent and writes the JSON/log to `MyDrive/RAGLedger/preflight/`, never
to the probe directory.

It must print `READY` and contain a `PROTECTED` doctor with successful
embedding, generation, NLI and trust-ledger probes. Exit codes are `0 READY`,
`1 DEGRADED`, `2 BLOCKED`; only `0` proceeds.

## 6. Integration pilot — 5,000 / 20 / 12

```bash
python scripts/run_trust_stream.py \
  --config config/colab_pilot_qwen.yaml \
  --corpus "$PILOT/corpus_subset.parquet" \
  --queries "$PILOT/queries_subset.jsonl" --qrels "$PILOT/qrels.tsv" \
  --run-purpose pilot --sample-mode zipf \
  --limit 5000 --steps 12 --targets 2 \
  --out "$PILOT_RUN" \
  --set embedding.revision=2bd46a25019aeea091fd42d1f0fd4801675cf699 \
  --set security.nli_revision=6c749ce3425cd33b46d187e45b92bbf96ee12ec7 \
  --set generation.revision="$QWEN_DIGEST" \
  --set generation.quantization="$QWEN_QUANT" \
  --set storage.root="$PILOT_RUN"
```

The manifest records `run_purpose: pilot`, the alternate model ID/backend/
revision/quantisation, query-pool size 20, 12 requested steps, 12 completed and
the distinct-query count. The 12 Zipf steps are **not** 12 unique queries and are
not research evidence.

## 7. Research experiment — 200K / 500 unique queries

Run only after the pilot is `READY` and has a successful manifest. Use the exact
pinned Llama revision and an adequate GPU:

```bash
python scripts/check_research_environment.py \
  --config config/nq_gpu.yaml --functional --json \
  --preflight-storage-root "$PREFLIGHT_PARENT/research" \
  --corpus "$RESEARCH/corpus_subset.parquet" \
  --queries "$RESEARCH/queries_subset.jsonl" --qrels "$RESEARCH/qrels.tsv" \
  --set embedding.revision=2bd46a25019aeea091fd42d1f0fd4801675cf699 \
  --set security.nli_revision=6c749ce3425cd33b46d187e45b92bbf96ee12ec7 \
  --set generation.backend=openai \
  --set generation.model_name=meta-llama/Llama-3.1-8B-Instruct \
  --set generation.revision=a09a35458c702b33eeacc393d103063234e8bc28 \
  --set generation.precision=bfloat16 \
  --set generation.base_url=http://127.0.0.1:8000/v1 \
  --set storage.root="$RESEARCH_RUN"

python scripts/run_trust_stream.py \
  --config config/nq_gpu.yaml \
  --corpus "$RESEARCH/corpus_subset.parquet" \
  --queries "$RESEARCH/queries_subset.jsonl" --qrels "$RESEARCH/qrels.tsv" \
  --run-purpose research --sample-mode no-repeat \
  --steps 500 --targets 5 --out "$RESEARCH_RUN" \
  --set embedding.revision=2bd46a25019aeea091fd42d1f0fd4801675cf699 \
  --set security.nli_revision=6c749ce3425cd33b46d187e45b92bbf96ee12ec7 \
  --set generation.backend=openai \
  --set generation.model_name=meta-llama/Llama-3.1-8B-Instruct \
  --set generation.revision=a09a35458c702b33eeacc393d103063234e8bc28 \
  --set generation.precision=bfloat16 \
  --set generation.base_url=http://127.0.0.1:8000/v1 \
  --set storage.root="$RESEARCH_RUN"
```

`--sample-mode no-repeat` samples query indices without replacement and refuses
to shorten a 500-step request if the pool is smaller. The manifest asserts
`query_pool_size: 500`, `stream_steps_completed: 500`,
`distinct_queries_evaluated: 500` and `repeated_query_evaluations: 0`.

The Zipf trust-lifecycle stream is a **separate experiment**:

```bash
python scripts/run_trust_stream.py ... \
  --run-purpose research_lifecycle --sample-mode zipf \
  --steps 500 --targets 5 --out "$LIFECYCLE_RUN"
```

It reports 500 steps and a smaller distinct-query count. Never call its 500
steps "500 unique queries". Its output directory is separate.

`--run-purpose research` still is **not** the complete attack × baseline
benchmark. It runs one trust stream. Person C's V0–V5 matrix, baselines,
answer-level ASR criterion, confidence intervals and multi-seed analysis remain
in `docs/PERSON_C_TODO.md`; answer-level ASR is **NOT MEASURED**.

## 8. Resume after a runtime disconnection

The runner writes `checkpoint.json` before and after every step and appends each
completed row to `stream.jsonl`. The index, provenance/answer stores and SQLite
trust ledger stay in the active `/content` run directory and are snapshotted to
Drive; only the Drive copies survive a runtime reset.

```bash
# /content still present
python scripts/run_trust_stream.py <the exact original arguments> --resume

# After a runtime reset, let the runner restore a matching Drive snapshot first
python scripts/run_trust_stream.py <the exact original arguments> \
  --backup-dir /content/drive/MyDrive/RAGLedger/backups/<run-id> \
  --restore --resume
```

`--resume` and `--restore` require the same full signature: Git commit, resolved
config, embedding/generator/NLI revisions, dataset hashes, seed, purpose, sample
mode, steps, targets, limit, source count and target query ids. A changed model
or dataset is rejected rather than mixed. Completed steps are read from
`stream.jsonl`; work resumes at the next step. The runner computes the signature
once, restores only a matching snapshot, then validates the restored active
directory before entering the resume path. It does not install a snapshot that
belongs to another experiment.

If the checkpoint says `in_flight_step` is not null, the previous process died
inside a step. The ledger may contain partial trust effects, so resume fails
closed; start a **new run directory** or restore a snapshot that predates the
in-flight step. A completed run (with `metrics.json`) is never overwritten.
`--skip-index` is no longer sufficient; use `--resume`. Checkpoint/stream row
count/SQLite/index disagreement is a refusal, not a repair.

## 9. Outputs and export

Each completed run contains `run_manifest.json`, `metrics.json`, `stream.jsonl`,
`trust_history.csv`, the provenance/answer/trust SQLite databases,
`checkpoint.json`, the saved FAISS index and optional figures. The manifest
records:

* Git commit and full command;
* resolved config and dataset SHA-256 identities;
* exact embedding/generator/NLI model IDs, revisions, backend, precision and
  quantisation (or `UNSPECIFIED`, never guessed);
* hardware, CUDA/PyTorch, GPU VRAM, ledger path, indexed passages;
* run purpose, query sampling, pool size, steps, distinct queries, targets;
* security status and readiness failures.

The notebook exports manifests, metrics, stream, history, checkpoint, figures
and consistent provenance/answer/trust SQLite backups to
`MyDrive/RAGLedger/exports/`. If `/content` was reset, the exporter falls back
to the latest completed Drive snapshot and records that provenance instead of
pretending the active directory still exists. The archive does not duplicate
the large index: the prepared index remains once under
`MyDrive/RAGLedger/backups/<run-id>/index` for a future runtime, and its digest
is recorded in the export metadata.

## 10. Hardware and limitations

* GPU is required for the notebook; it aborts without CUDA.
* The pilot has been tested only offline locally; **not in Colab**.
* Local validation used `pyarrow 26.0.0`; the Colab target pin is
  `pyarrow==17.0.0`. The pinned Colab stack remains unverified until the
  notebook is actually executed there.
* `faiss-cpu` means CPU FAISS indexing. No GPU FAISS claim is made.
* Qwen Ollama q4_K_M is an alternate pilot model, not Llama research.
* Llama-3.1-8B research requires a larger GPU than a free T4, plus gated model
  access and a user-controlled vLLM download.
* No real NQ results, no pilot result and no Colab execution are claimed.
* Answer-level ASR remains **NOT MEASURED**.
