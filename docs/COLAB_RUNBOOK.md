# Running TRACE-RAG end to end on Colab (GPU)

The notebook version of this runbook is [`notebooks/trace_rag_colab.ipynb`](../notebooks/trace_rag_colab.ipynb)
(upload it at <https://colab.research.google.com> → File → Upload notebook). This
file is the same run as copy-paste commands, plus what to expect and what to
report.

Set **Runtime → Change runtime type → GPU** before anything else.

## 1. Install

```bash
git clone --depth 1 https://github.com/vaishnavissl1/-Adaptive-Trust-Based-Framework-for-Detecting-and-Preventing-Knowledge-Base-Poisoning-in-RAG-Systems.git /content/trace-rag
cd /content/trace-rag
pip install -q -e ".[all,data]"
python scripts/check_setup.py          # CUDA should now say OK, ollama OK after step 3
```

If the repo is private, use `https://<TOKEN>@github.com/...` with a token that
has `repo` scope. Colab already ships a CUDA build of PyTorch, and
`pip install -e ".[all,data]"` leaves it alone because `torch>=2.1` is satisfied.

## 2. Data

Development corpus — 200k passages, keeps every gold passage for 500 test
queries (764 MB download, resumable):

```bash
python scripts/download_data.py --dataset nq --out data/nq --subset 200000 --queries 500
```

Full corpus — all 2.68M passages, for the `full_nq.yaml` configuration:

```bash
python scripts/download_data.py --dataset nq --out data/nq --queries 500
# then use --config config/full_nq.yaml everywhere below (IVF-PQ index, ~2-4 h on a T4)
```

## 3. Answer model (Ollama)

```bash
apt-get -qq install -y zstd   # the installer unpacks a zstd archive; Colab does not ship zstd
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >/tmp/ollama.log 2>&1 &
ollama pull qwen2.5:7b        # ~4.7 GB; llama3.1:8b also works
```

Skipping `zstd` is the first thing that goes wrong on Colab: the installer stops
with `This version requires zstd for extraction`.  Colab has no systemd either,
so the daemon is started by hand; a runtime restart kills it and it has to be
started (not re-downloaded) again.

Check the model answers, then check it through the project's own backend:

```bash
curl -s http://127.0.0.1:11434/api/tags | head -c 400
python -c "
import sys; sys.path.insert(0, 'src')
from trace_rag.generation.llm import OllamaLLM
llm = OllamaLLM('qwen2.5:7b')
r = llm.generate('Answer in one short sentence: who designed the Eiffel Tower?', max_tokens=48)
print(round(r.latency_ms), 'ms |', r.text)
"
```

vLLM instead: `pip install vllm && vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`,
then use `--set generation.backend=openai --set generation.base_url=http://localhost:8000/v1`.

## 4. Quick validation (~5-10 min)

```bash
python scripts/run_trust_stream.py \
  --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl \
  --qrels data/nq/qrels.tsv \
  --limit 20000 --steps 6 --targets 2 --out runs/nq/quick \
  --set generation.backend=ollama --set generation.model_name=qwen2.5:7b
```

## 5. Full 200k run (~25-45 min on a T4)

```bash
python scripts/run_trust_stream.py \
  --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl \
  --qrels data/nq/qrels.tsv \
  --steps 40 --targets 5 --out runs/nq/trust_stream \
  --set generation.backend=ollama --set generation.model_name=qwen2.5:7b
```

Add `--skip-index` (with the **same** `--out`) to reuse the saved index on a
second run in the same session.

## 6. What the run produces

`runs/nq/trust_stream/`:

| File | What it is |
|---|---|
| `metrics.json` | smoke-test metrics: poison retrieval/citation rate, quarantine counts, verifier verdict counts, LLM calls and latency per query |
| `stream.jsonl` | one row per stream step: answer, bands, per-document verdicts, newly quarantined ids |
| `trust_history.csv` | every trust observation (B6 export) |
| `figures/` | one trust-over-time plot per document/source, plus `quarantine_timeline.png` when anything was blocked |
| `trust.sqlite3` | the ledger itself: entities, transitions, admin review queue |

Flag reference (the details matter for reading the numbers):

* `--targets N` — how many target questions get a poisoned copy of their gold
  passage. Five in one minute is also what trips the burst detector
  (`burst_min_docs=4`), so the Sybil/cold-start discount is exercised.
* `--steps N` — queries in the Zipf stream; target questions recur, which is
  what lets a repeated verdict reach the two refutations needed to quarantine.
* `--quarantine-refutations N` — refutations before QUARANTINED (plan default 2).
* `--no-hierarchical` — V2 ablation: document-only trust, no family/source prior
  and no inheritance.
* `--force-medium` — debugging only: escalates *every* passage so the verifier
  always runs. It also verifies off-topic passages that real banding keeps at
  LOW; measured on the bundled mini corpus with a real model, that refuted and
  quarantined clean passages by mistake. Do not quote numbers from this mode.

## 7. How to read it

* The poison **should** be retrieved — that is the attack landing at scale.
  Defence shows up as `poison_quarantined ≥ 1`, falling `t_eff` for the poison
  chunks, and `poison_citation_rate` at or near 0.
* `clean_false_quarantine` should be 0.
* If the poison is never escalated, the trust layer never sees it: that is the
  detector's coverage (Person A's heuristic fallback until a labelled scorer is
  trained), not a trust-layer failure.
* A poison that is refuted once shows up as `MONITORED` with `t_eff` down from
  0.5 to ~0.35. Two refutations are needed for `QUARANTINED`, and it is the
  *repeated* target queries in the stream that earn the second one; a short
  stream may stop at `MONITORED`.
* These are smoke-test numbers. ASR, the V0-V5 matrix, baselines and confidence
  intervals are Person C's runner (`docs/HANDOVER_B_TO_C.md` lists what it needs).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda available: False` | runtime is not GPU — change it and re-run from cell 0 |
| `cannot reach http://localhost:11434` | daemon died with the runtime: re-run the Ollama cell |
| `This version requires zstd for extraction` | `apt-get -qq install -y zstd`, then re-run the Ollama cell (already handled in the notebook) |
| `ollama pull` slow / disk | ~5 GB; `qwen2.5:3b` is the lighter fallback |
| `CUDA out of memory` while indexing | `--set embedding.batch_size=32` |
| `CUDA out of memory` from the model | smaller model (the embedder, NLI and LLM share the GPU) |
| download interrupted | re-run; it resumes |
| `needs pyarrow` | `pip install pyarrow` (already in the `data` extra) |
