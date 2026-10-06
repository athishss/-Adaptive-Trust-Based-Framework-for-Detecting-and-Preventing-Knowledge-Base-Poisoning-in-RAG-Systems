# TRACE-RAG on Google Colab TPU

Notebook: [`notebooks/trace_rag_colab_tpu.ipynb`](../notebooks/trace_rag_colab_tpu.ipynb). Upload it to Colab or open it from the repository, select **Runtime → Change runtime type → TPU**, then run all cells in a fresh runtime.

## What the notebook does

1. Clones the fixed Arena branch without embedding credentials in notebook cells.
2. Probes `torch_xla` in a short-lived process and installs the project, Parquet, plotting, and test dependencies **without replacing Colab's torch/torch_xla pair**.
3. Runs `pytest -q -ra` and retains the test log.
4. Runs `scripts/run_trust_stream.py` on the bundled mini corpus with MiniLM embeddings on TPU/XLA and DeBERTa NLI on TPU/XLA. The default answer backend is `StubLLM` so the flow works without API credentials; its answers are not answer-quality evidence.
5. Optionally downloads the BEIR Natural Questions source files, prepares a gold-preserving 20k/100-query subset, and runs the same smoke stream using `config/nq_tpu.yaml`.
6. Measures one CPU and one TPU embedding pass in isolated processes, separating model load, first-batch warm-up, and steady-state timing. The comparison is explicitly indicative because XLA uses static padding and synchronization.
7. Saves resolved configuration, raw metrics, stream output, trust history, logs, the timing probe, a presentation CSV, run metadata, and a ZIP archive.

Set `RUN_NQ_SUBSET = True` in the notebook only if you have enough disk, network time, and accelerator budget. The subset builder first downloads the full source corpus. The NQ stream remains a component smoke, not the complete attack × baseline × seed benchmark.

## Accelerator scope and timing

The current TPU config sends transformer embeddings and (when selected) the NLI cross-encoder through PyTorch/XLA. NumPy retrieval/indexing, SQLite, parsing, signals, and most trust operations remain on CPU. Hugging Face generation can optionally select XLA via `generation.device=tpu`, but autoregressive generation is experimental and may be slow; the default stub avoids silently making a model-quality claim. First-run graph compilation may dominate elapsed time, so retain the timing output and distinguish compile/warm-up from steady state before comparing devices.

Use the actual `embedding_device` and `verifier_device` fields in `metrics.json` to verify that a run used XLA. The notebook asserts both start with `xla` for the mini run. Merely selecting a TPU runtime is not proof that the model used it.

## Reproducing a command-line mini run

After installing the dependencies and selecting a TPU runtime, the core command used by the notebook is equivalent to:

```bash
TRACE_RAG_NLI_DEVICE=tpu TRACE_RAG_NLI_MAX_LENGTH=256 \
python scripts/run_trust_stream.py \
  --config config/default.yaml \
  --corpus examples/mini_corpus.jsonl \
  --queries examples/mini_questions.json \
  --steps 12 --targets 2 --seed 20260921 \
  --out runs/colab_tpu/mini \
  --set embedding.backend=huggingface \
  --set embedding.model_name=sentence-transformers/all-MiniLM-L6-v2 \
  --set embedding.device=tpu \
  --set embedding.batch_size=4 \
  --set embedding.max_length=128 \
  --set trust.nli_mode=nli \
  --set generation.backend=stub
```

For a small local answer model, change `generation.backend` to `huggingface`, set `generation.model_name=Qwen/Qwen2.5-0.5B-Instruct` and `generation.device=tpu`. This may consume additional TPU memory and time; it is not a substitute for a validated answer model or benchmark.

## Results and limitations

Each run writes `resolved_config.yaml`, `metrics.json`, `stream.jsonl`, `trust_history.csv`, `trust.sqlite3`, `runner.log`, and figures when Matplotlib is installed. The parent notebook folder also contains `pytest_output.txt`, `presentation_summary.csv`, `run_metadata.json`, and a ZIP archive. Optional Drive persistence is in the last notebook cell.

This environment did not have a live TPU, torch/Transformers, FAISS, or a real dataset, so the TPU path has not been executed here. Treat a successful user-side notebook run as the first hardware validation for that environment. The mini corpus is a plumbing regression only; do not cite its rates as benchmark estimates. TRACE-RAG remains a research prototype, not production-ready for company use; see [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
