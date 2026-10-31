# Real-model Natural Questions run

This is the protocol for the intended research-grade run: real NQ passages,
Contriever embeddings, FAISS retrieval, an 8B instruction model and the DeBERTa
NLI verifier. **It has not been executed in this repository's verification
environment.** Everything below is a runbook, not a report of results. Do not
copy smoke-test numbers into a paper and label them as this experiment.

The deterministic offline smoke test (`scripts/run_chronological_eval.py`) and
the mini trust stream (`scripts/run_trust_stream.py --config
config/default.yaml ...`) are separate, clearly labelled scopes.

> **Status: NOT EXECUTED.** This document is a runbook, not a report of results.
>
> * The configured scope is a **200,000-passage / 500-query subset**, not the
>   full 2.68M-passage Natural Questions corpus.
> * No real-model run has been completed in this repository; there are no
>   real-model results to report.
> * Answer-level ASR is **not measured** by the trust-stream runner. It reports
>   retrieval, poison exposure and poison citation only; those are not answer
>   success.
> * A real attack × baseline evaluation matrix on Natural Questions remains
>   future work; only the fixture-smoke matrix exists here.

---

## 1. What the run establishes

With the commands below you can measure, on real passages:

* whether poisoning reaches the retrieved context and the generated answer;
* how often the cheap signals escalate poison versus clean content;
* whether independent corroboration resolves the escalation (SUPPORT / REFUTE /
  NEUTRAL) under the NLI model;
* how trust evolves at document, family and source level;
* whether quarantine happens, and whether earlier answers are retroactively
  flagged;
* the cost side: LLM calls per query, verifier latency and queue behaviour.

It does **not** by itself produce a full attack × baseline ASR matrix on NQ.
The fixture matrix (`runs/final_smoke`) is the only matrix runner in this
repository. A real-NQ matrix would need a labelled-row generator and a
real-data runner; that remains a limitation (see §8).

---

## 2. Prerequisites

| Requirement | Detail |
|---|---|
| Python | 3.10+; 3.11 or 3.12 is the safer choice for the scientific stack |
| GPU | CUDA GPU with enough VRAM for Contriever + DeBERTa NLI + the generator; exact fit depends on the model and batch size |
| Disk | NQ corpus + queries + qrels + model weights; the full BEIR NQ corpus is several GB |
| Packages | `pip install -e ".[all,data]"` (FAISS, torch, transformers, pyarrow, PDF/HTML parsers) |
| Generator | vLLM serving an OpenAI-compatible endpoint, or Ollama, or a local HF model |
| Network | First run downloads Contriever, DeBERTa NLI and the generator weights |

The environment used for the committed smoke results had none of the optional
backends installed. That is why `trace-rag doctor` reports `DEGRADED` there; it
is not evidence that the real backends work.

### Hardware note

`config/full_nq.yaml` uses FAISS IVF-PQ because a flat index over 2.68M × 768
float32 vectors needs roughly 8.2 GB of RAM before metadata. `nq_gpu.yaml` uses
a flat index and is intended for a subset (for example 200k passages). Start
with the subset; do not begin with the full corpus.

---

## 3. Download and prepare NQ

```bash
python scripts/download_data.py --dataset nq --out data/nq \
  --subset 200000 --queries 500
```

The downloader is resumable and writes:

```
data/nq/corpus.parquet           # full BEIR corpus (streamed download)
data/nq/queries.parquet          # BEIR queries
data/nq/qrels.tsv                # BEIR qrels (test split)
data/nq/corpus_subset.parquet    # 200k passages, every gold passage kept
data/nq/queries_subset.jsonl     # the 500 queries covered by the subset
```

`--subset` is not a random sample: it keeps every gold passage for the sampled
queries so retrieval is not broken by construction. `--queries` controls how
many test queries the subset must cover.

If the download is interrupted, rerun the same command. To rebuild the subset
from files already on disk without downloading again, add `--skip-download`.

### Data licence

Natural Questions is distributed by Google; BEIR repackages it. Review the
dataset card and the BEIR repository's licence/citation terms before
redistributing passages, queries or derived text. Do not commit `data/` or model
weights to this repository. The bundled `examples/mini_*.jsonl` files are
synthetic fixtures, not NQ samples.

---

## 4. Configure the real backends

`config/nq_gpu.yaml` already selects:

```yaml
embedding:
  backend: huggingface
  model_name: facebook/contriever
  device: auto
index:
  backend: faiss
  faiss_kind: flat
generation:
  backend: openai
  model_name: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
security:
  enabled: true
  verifier: nli
  queue_size: 1000
  drain_queue: true
  fail_open_without_verifier: false
trust:
  quarantine_refutations: 2
  reject_refutations: 3
```

Start the generator before querying. With vLLM:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Or, for a laptop without vLLM:

```bash
ollama run llama3.1:8b
# then override:
# --set generation.backend=ollama --set generation.model_name=llama3.1:8b \
# --set generation.base_url=http://localhost:11434
```

The NLI verifier loads `cross-encoder/nli-deberta-v3-base` on first use. It
needs `torch` and `transformers`; if it cannot load, the code logs a warning and
falls back to lexical. **A lexical fallback means the run is not the NLI
experiment.** Check the mode before scoring:

```bash
trace-rag --config config/nq_gpu.yaml doctor
```

Required for a reportable run:

```
status: PROTECTED
verifier_mode: nli
trust_persistent: true
queue_enabled: true
research_ready: true
```

If `doctor` says `DEGRADED`, fix the backend before running the experiment.

---

## 5. Ingest and index

```bash
trace-rag --config config/nq_gpu.yaml ingest-beir \
  --corpus data/nq/corpus_subset.parquet

trace-rag --config config/nq_gpu.yaml index \
  --save-every 20000
```

`index` is resumable: already-indexed passages are skipped, and `--save-every`
checkpoints the FAISS index. Use `--limit N` to work in sittings. Do not delete
the provenance database between sittings; the index and the store must stay in
sync.

Check the store and index before the stream:

```bash
trace-rag --config config/nq_gpu.yaml stats
```

---

## 6. Run the real trust stream

```bash
python scripts/run_trust_stream.py \
  --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl \
  --qrels data/nq/qrels.tsv \
  --steps 40 --targets 3 --seed 20260921 \
  --out runs/nq/trust_stream
```

What the script does:

1. ingests the clean corpus with simulated contributors;
2. indexes it;
3. chooses, for each target question, a passage that actually carries the gold
   answer (from `qrels.tsv` when present, otherwise the retrieved passage that
   contains it);
4. corrupts that passage with `entity_swap` or `negation`, and ingests the
   poison as a burst from one fresh attacker source;
5. replays a Zipf-repeated query stream through the full pipeline with the
   trust ledger, verifier, policy and queue;
6. writes metrics, per-step records, trust history and figures.

Useful diagnostic switches (not for the headline run):

```bash
# force every retrieved passage into MEDIUM to exercise the verifier
# --force-medium

# disable source/family trust aggregation (ablation)
# --no-hierarchical

# change the quarantine threshold
# --quarantine-refutations 2
```

`--skip-index` reuses an existing index during iteration.

### Outputs

```
runs/nq/trust_stream/
  metrics.json          poison retrieval/citation/quarantine, false quarantine,
                        band counts, verification outcomes, trust snapshots,
                        LLM calls, latency
  stream.jsonl          one record per query: retrieved/context/cited ids,
                        bands, verification results and evidence ids
  trust_history.csv     per-document trust history
  figures/              trust-dynamics and quarantine-timeline plots
```

Read `metrics.json` before making any claim. In particular:

* `poison_citation_rate` counts poison passages cited by the generated answer;
* `poison_quarantined` and `clean_false_quarantine` are document-level;
* `verifications` separates SUPPORT / REFUTE / NEUTRAL;
* `poison_trust` shows the final `t_eff` and status per poison document;
* `verifier_mode` and `llm` must say what actually ran.

The script does **not** compute an attack-success rate; it reports exposure and
citation of the injected documents. Answer-level ASR is **NOT MEASURED** by
this runner. Do not call `poison_citation_rate` ASR.
For a real ASR matrix, build a runner that scores generated answers against the
gold answer and compares all eight baselines on the same queries.

---

## 7. Minimal acceptance checks for a real run

Before treating a run as usable:

1. `trace-rag doctor` reports `PROTECTED`, `nli`, persistent ledger, queue on.
2. The index size equals the number of indexed chunks in `stats`.
3. `metrics.json` shows `verifier_mode: nli`, not `lexical`.
4. At least one poison document is quarantined **and** at least one later
   target query retrieves the clean answer instead. If no poison is detected,
   report that; do not lower thresholds until a number appears.
5. `clean_false_quarantine` is reported, not hidden. A defence that quarantines
   clean content is a false positive even if ASR is zero.
6. LLM calls and latency are reported with the hardware and model names.
7. The exact command, commit, config, seed, dataset subset and environment are
   saved alongside the results.

A single seed is a smoke run. The fixture statistics module supports bootstrap
confidence intervals and McNemar's exact test; use multiple seeds and paired
comparisons for any claim of improvement.

---

## 8. What is still missing for a full NQ benchmark

* No real-NQ matrix runner for the seven attack families × eight baselines.
* No real-NQ labelled-row generator for training the calibrated scorer; the
  stream uses the heuristic scorer unless a trained `scorer.joblib` is provided.
* No measured real-model ASR, detection FPR, detection delay, remediation
  recall or cost numbers in this repository.
* The lexical verifier is not a substitute for NLI. Its false positives and
  false negatives are documented in
  [`ENGINEERING_REPORT.md`](ENGINEERING_REPORT.md).
* No multi-seed real run or confidence intervals have been produced.

Report these as limitations, not as completed experiments.
