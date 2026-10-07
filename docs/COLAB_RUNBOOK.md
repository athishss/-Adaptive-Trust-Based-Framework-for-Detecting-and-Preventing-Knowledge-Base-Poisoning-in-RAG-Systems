# Google Colab runbooks

## TPU / PyTorch XLA

Use [`notebooks/trace_rag_colab_tpu.ipynb`](../notebooks/trace_rag_colab_tpu.ipynb) with [`COLAB_TPU_RUNBOOK.md`](COLAB_TPU_RUNBOOK.md). It targets the full BEIR NQ corpus, TPU/XLA embeddings and NLI, and a real Ollama generator running on the Colab CPU/GPU. Select a **TPU** runtime before executing it. This path is experimental and has not been hardware-validated in the development sandbox; the notebook checks the actual XLA device and fails fast if one is unavailable. Read the explicit provenance and controlled-attack caveats before using its output.

## Full BEIR NQ on a Colab T4

Use [`notebooks/trace_rag_colab_t4.ipynb`](../notebooks/trace_rag_colab_t4.ipynb) with [`COLAB_T4_RUNBOOK.md`](COLAB_T4_RUNBOOK.md). It targets the full 2.68M-passage BEIR NQ corpus, Contriever and NLI on CUDA, CPU FAISS IVF-PQ, and a real Ollama model. It samples 500 qrels-backed queries and is an integration run—not a complete attack/baseline/seed benchmark. Read the hardware and data-provenance caveats before using its output.

## Other CUDA walkthroughs

For the existing development GPU walkthrough, see [`RUNNING_ON_GPU.md`](RUNNING_ON_GPU.md). The legacy [`notebooks/trace_rag_colab.ipynb`](../notebooks/trace_rag_colab.ipynb) is GPU-only and uses its existing component-smoke scope; it is separate from both the full-NQ T4 notebook and the TPU notebook.

None of these notebooks or runbooks establishes production readiness or research headline results. See [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) and [`REQUIREMENTS.md`](REQUIREMENTS.md).
