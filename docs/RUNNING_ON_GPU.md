# Running on a CUDA GPU with Anaconda

Written for one NVIDIA GPU on Linux or WSL2. Every command is meant to be
copy-pasted in order.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate trace-rag
```

PyTorch is **not** pinned in that file, because the right build depends on your
driver. Check your driver first:

```bash
nvidia-smi          # note the CUDA version in the top-right corner
```

Then install the matching PyTorch build from the official selector at
<https://pytorch.org/get-started/locally/>. For a CUDA 12.4 driver that is:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line must print `True` and your GPU's name. If it prints `False`, the
PyTorch build does not match your driver — reinstall with the right index URL
rather than continuing.

Finally install this package:

```bash
pip install -e ".[all,data]"
pytest -q                    # 148 tests; the model-backend tests now run too
```

## 2. Get the data

```bash
python scripts/download_data.py --dataset nq --out data/nq --subset 200000 --queries 500
```

* Downloads the Natural Questions corpus (764 MB), queries and test qrels from
  Hugging Face. Resumable — re-run it if your connection drops.
* `--subset 200000` builds a 200k-passage development corpus that **keeps every
  gold passage** for 500 sampled test queries. A random slice would drop the
  gold passages and make retrieval look broken for the wrong reason.
* Drop `--subset` to work with all 2,681,468 passages.

Other datasets: `--dataset hotpotqa`, `--dataset msmarco`.

## 3. Serve the answer model

Option A, vLLM (fastest, needs a GPU with ~16 GB free for Llama-3.1-8B in bf16):

```bash
pip install vllm
huggingface-cli login                  # Llama 3.1 is gated: request access first
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype bfloat16
```

Option B, Ollama (works on smaller GPUs, quantised):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.1:8b
```

then in `config/nq_gpu.yaml` switch the generation block to the Ollama one that
is already written there as a comment.

Option C, no gated access: use `Qwen/Qwen2.5-7B-Instruct`, which is Apache-2.0
and needs no approval. Change `generation.model_name` only.

## 4. Ingest, index, ask

```bash
trace-rag --config config/nq_gpu.yaml ingest-beir --corpus data/nq/corpus_subset.parquet
trace-rag --config config/nq_gpu.yaml index
trace-rag --config config/nq_gpu.yaml query "who designed the eiffel tower?"
trace-rag --config config/nq_gpu.yaml stats
```

Rough expectations on one modern GPU, 200k passages:

| Step | Time | Notes |
|---|---|---|
| Ingest | 1–2 min | CPU only; `family_backend: exact` keeps memory flat |
| Index (embedding) | 5–15 min | this is what the GPU is for |
| Query | under a second | plus the LLM's own latency |

The full 2.68M corpus is roughly 13× the indexing time, and needs
`faiss_kind: ivfpq` (the commented block in the config) — a flat index of
2.68M × 768 float32 vectors wants about 8.2 GB of RAM.

## 5. Train the scorer

Person A's detector is trained from labelled rows that Person C's harness
produces. Once you have `labelled_rows.jsonl`:

```bash
trace-rag --config config/nq_gpu.yaml train-scorer --rows runs/nq/labelled_rows.jsonl
```

then set `scorer.model_path: runs/nq/scorer.joblib` in the config. Until then
the pipeline uses the heuristic fallback and says so in `stats`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | PyTorch build does not match the driver; reinstall with the index URL for your CUDA version |
| `CUDA out of memory` while indexing | lower `embedding.batch_size` (128 → 32) |
| `CUDA out of memory` from the LLM | use Ollama (quantised) or a 7B model; 8B in bf16 needs ~16 GB |
| `requires accelerate` | only when `device: auto`; use `cuda` or `pip install accelerate` |
| `401`/`403` downloading Llama | the model is gated: request access on its Hugging Face page, then `huggingface-cli login` |
| Download stops part way | re-run `download_data.py`; it resumes |
| `needs pyarrow` | `pip install pyarrow` (or use the `.jsonl` corpus) |
| Index seems empty after restart | run `trace-rag index` again, or make sure the earlier run finished and saved |

## Reproducibility notes

* Decoding is greedy (`temperature: 0`) with a fixed seed.
* The same config and seed give the same retrieval order on the same machine.
* GPU floating-point differences mean embeddings can differ in the last decimal
  between machines; ranking is stable because ties break on passage id.
