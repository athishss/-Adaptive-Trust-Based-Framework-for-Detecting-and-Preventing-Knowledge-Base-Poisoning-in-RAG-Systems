# Google Colab runbooks

## TPU / PyTorch XLA

Use [`notebooks/trace_rag_colab_tpu.ipynb`](../notebooks/trace_rag_colab_tpu.ipynb) with [`COLAB_TPU_RUNBOOK.md`](COLAB_TPU_RUNBOOK.md). Select a **TPU** runtime before executing it. This path is experimental and has not been hardware-validated in the development sandbox; the notebook checks the actual XLA device and fails fast if one is unavailable.

## CUDA GPU

For a CUDA runtime, see [`RUNNING_ON_GPU.md`](RUNNING_ON_GPU.md). The legacy [`notebooks/trace_rag_colab.ipynb`](../notebooks/trace_rag_colab.ipynb) is GPU-only; it is not the TPU notebook. Its runs are component smokes unless you explicitly use a real model and a complete benchmark protocol.

Neither notebook/runbook establishes production readiness or research headline results. See [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) and [`REQUIREMENTS.md`](REQUIREMENTS.md).
