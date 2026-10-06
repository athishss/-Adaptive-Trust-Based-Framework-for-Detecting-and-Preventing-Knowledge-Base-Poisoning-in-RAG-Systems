# TRACE-RAG full BEIR NQ run on Google Colab TPU

Notebook: [`notebooks/trace_rag_colab_tpu.ipynb`](../notebooks/trace_rag_colab_tpu.ipynb). Open it in Colab, select **Runtime → Change runtime type → TPU**, restart, then run from the top. It is designed for a complete BEIR Natural Questions corpus integration run, not a bundled mini-corpus demonstration.

## What the notebook runs

1. Clones the fixed Arena branch and verifies a working PyTorch/XLA TPU in a short-lived process.
2. Installs the project, FAISS CPU, Parquet, plotting, test, and Transformers dependencies without explicitly replacing Colab's `torch`/`torch_xla` pair.
3. Runs `pytest -q -ra`, including the Person A/B integration tests and Person C attack, baseline, metrics, statistics, and matrix-helper tests.
4. Starts Ollama, downloads `qwen2.5:3b`, and verifies a real generation request. Ollama runs on the Colab VM's CPU or supported GPU; it **does not run on the TPU**.
5. Downloads BEIR NQ and verifies the complete **2,681,468-passage** `corpus.parquet`. The downloader samples 500 genuine qrels-backed query records into `queries_subset.jsonl` without materializing the corpus in Python lists. It does not request `--subset`.
6. Runs `scripts/run_trust_stream.py` over the full corpus using Contriever embeddings and NLI verification on TPU/XLA, FAISS IVF-PQ and SQLite on CPU, the real Ollama answer model, Person B trust/policy/remediation, and Person C's controlled attack helpers. Index checkpoints are saved every 50,000 newly embedded chunks.
7. Runs each implemented Person C baseline once against one authentic NQ retrieval result. This verifies that all baseline APIs execute with real retrieved text and includes BEIR qrels retrieval metrics; it is only a one-query diagnostic, not a matched attack/baseline benchmark.
8. Saves configs, raw stream metrics, logs, trust history/figures, test output, the one-query baseline diagnostic, and an archive. The archive intentionally excludes the NQ corpus and downloaded model caches.

The stream has 500 steps sampled/repeated from 500 real queries; it is not a run over every NQ query. It uses 10 controlled attack targets, each derived from a real qrels-relevant NQ passage. It does not run every attack family across every baseline and independent seed.

## Data provenance and the “no synthetic” requirement

BEIR Natural Questions contains real passages, queries, and qrels, but does **not** contain contributor identities, original ingestion timestamps, or real poisoning incidents. The notebook therefore avoids fabricating a roster of clean contributors: `--source-from-title` derives page-level source IDs from the authentic BEIR titles, and `--live-timestamps` records the wall-clock start of the import. This is page-level provenance, not author/contributor identity; BEIR cannot provide original source ages.

Poison documents are controlled generated mutations of authentic qrels-relevant passages, and the attacker source is a controlled test identity. These are necessary to exercise the poisoning defenses, but they are not organic incident data. The report and notebook label them explicitly. If “no synthetic” means no generated/modified attack documents or simulated attacker identity whatsoever, this dataset cannot satisfy that requirement and Person C's attack-defense evaluation cannot run as requested; supply authentic incident payloads and source/provenance metadata instead. Do not present these controlled attacks as real incidents or source assignments as real contributor identities.

## Runtime and storage

Full-corpus ingest/index may take hours and use substantial RAM and disk. Reserve space for the source parquet, SQLite provenance/answer/trust files, FAISS checkpoint, Hugging Face models, the Ollama Qwen2.5 3B model, DistilGPT-2 for the diagnostic perplexity baseline, and logs. Actual use depends on row chunking and runtime resources; no speed or capacity guarantee is made.

The TPU configuration sends Contriever embeddings and the NLI cross-encoder to PyTorch/XLA. FAISS IVF-PQ, parsing, SQLite, retrieval bookkeeping, signal computation, and most policy work remain on CPU. Ollama does not use XLA. Verify the resolved `embedding_device` and `verifier_device` in `nq_full/metrics.json`; selecting a TPU runtime alone is not proof of TPU execution.

Indexing uses `--save-every 50000`. If the process is interrupted **during indexing after clean-corpus ingestion completed**, the script can load the checkpoint and resume missing vectors with `--skip-index`. Do not blindly rerun into an existing output directory after interruption during poison/stream evaluation: that replays the attack and query stream against existing state. Start a new `--out` directory for a fresh experiment.

## Reproducing the full stream command

After preparing the full NQ corpus, starting Ollama, and selecting a TPU runtime:

```bash
TRACE_RAG_NLI_DEVICE=tpu TRACE_RAG_NLI_MAX_LENGTH=256 \
python scripts/run_trust_stream.py \
  --config config/nq_tpu.yaml \
  --corpus data/nq/corpus.parquet \
  --queries data/nq/queries_subset.jsonl \
  --qrels data/nq/qrels.tsv \
  --steps 500 --targets 10 --seed 20260921 \
  --source-from-title --live-timestamps --save-every 50000 \
  --out runs/nq/full_colab \
  --set embedding.batch_size=8 \
  --set trust.nli_mode=nli \
  --set generation.backend=ollama \
  --set generation.model_name=qwen2.5:3b \
  --set generation.max_tokens=96 \
  --set generation.timeout_s=180
```

Use `scripts/download_data.py --dataset nq --out data/nq --queries 500` to retain the full corpus and generate the query sample. Do **not** pass `--subset` for the full-corpus run.

## Results and limitations

Each run writes `resolved_config.yaml`, `metrics.json`, `stream.jsonl`, `trust_history.csv`, the trust database, and figures when Matplotlib is available. The notebook's parent run directory contains the test log, `baseline_smoke.json`, metadata, CSV summary, Ollama log, and ZIP archive.

A successful notebook run establishes that the integrated code path ran on that specific Colab runtime. The stream metrics are still smoke/integration diagnostics; the one-query baseline test is not a research comparison. A full attack × baseline × seed protocol with matched per-query outputs, independent seeds, validated labels, confidence intervals, and statistical comparisons remains outstanding (see [`PERSON_C_TODO.md`](PERSON_C_TODO.md)).

This checkout has not been run on a live TPU or against a downloaded full corpus/Ollama model in the development sandbox. Do not claim that it has. TRACE-RAG remains a research prototype, not production-ready; see [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
