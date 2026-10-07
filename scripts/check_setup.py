#!/usr/bin/env python3
"""Check that this machine can run TRACE-RAG, and say exactly what is missing.

    python scripts/check_setup.py

Prints one line per check with OK / MISSING / WARN and a fix for anything that
is not OK.  Exit code 0 means the core pipeline will run (CPU, offline
backends); GPU and model items are reported but never fail the check.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import sys
from pathlib import Path

OK, MISSING, WARN = "OK     ", "MISSING", "WARN   "
failures: list = []


def line(status: str, name: str, detail: str = "", fix: str = "") -> None:
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if fix:
        print(f"          fix: {fix}")
    if status == MISSING:
        failures.append(name)


def check_module(name: str, label: str, fix: str, required: bool = True, version_attr: str = "__version__"):
    try:
        module = importlib.import_module(name)
    except ImportError:
        line(MISSING if required else WARN, label, "not installed", fix)
        return None
    version = getattr(module, version_attr, "?")
    line(OK, label, str(version))
    return module


def main() -> int:
    logging.getLogger("faiss").setLevel(logging.ERROR)     # its loader chatter is not useful here
    logging.getLogger("faiss.loader").setLevel(logging.ERROR)
    print(f"TRACE-RAG setup check\npython {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}\nworking directory: {Path.cwd()}\n")

    if sys.version_info < (3, 10):
        line(MISSING, "python >= 3.10", platform.python_version(),
             "conda create -n trace-rag python=3.11")
    else:
        line(OK, "python >= 3.10", platform.python_version())

    # the package itself
    try:
        import trace_rag

        line(OK, "trace_rag package", trace_rag.__version__)
    except ImportError:
        line(MISSING, "trace_rag package", "not importable",
             'run this from the project folder after: pip install -e ".[all,data]"')

    check_module("numpy", "numpy", "pip install numpy")
    check_module("sklearn", "scikit-learn", "pip install scikit-learn")
    check_module("pydantic", "pydantic", "pip install pydantic")
    check_module("yaml", "PyYAML", "pip install pyyaml", version_attr="__version__")
    check_module("joblib", "joblib", "pip install joblib")
    check_module("pyarrow", "pyarrow (reads BEIR Parquet)", "pip install pyarrow",
                 required=False)
    check_module("faiss", "faiss (large corpora)", "conda install -c conda-forge faiss-cpu",
                 required=False, version_attr="__version__")

    torch = check_module("torch", "torch (real embeddings)",
                         "see https://pytorch.org/get-started/locally/ and match your CUDA version",
                         required=False)
    if torch is not None:
        if torch.cuda.is_available():
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            free_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            line(OK, "CUDA", f"{names} ({free_gb:.1f} GB)")
            if free_gb < 12:
                line(WARN, "GPU memory", f"{free_gb:.1f} GB",
                     "Llama-3.1-8B in bf16 needs ~16 GB; use Ollama (quantised) or a 7B model")
        else:
            line(WARN, "CUDA", "torch cannot see a GPU",
                 "install the PyTorch build that matches `nvidia-smi`, or set device: cpu in the config")
    check_module("transformers", "transformers (real embeddings)",
                 "pip install transformers", required=False)

    # PyTorch/XLA is supplied with compatible TPU runtimes; do not install a
    # random torch_xla wheel over Colab's torch/XLA version pair.
    try:
        import torch_xla.core.xla_model as xm
        xla_device = xm.xla_device()
        if str(xla_device).startswith("xla"):
            line(OK, "PyTorch/XLA TPU", str(xla_device))
        else:
            line(WARN, "PyTorch/XLA TPU", f"no TPU device ({xla_device})",
                 "select a TPU runtime, or omit embedding.device=tpu")
    except ImportError:
        line(WARN, "PyTorch/XLA TPU", "not installed (optional)",
             "use a TPU runtime with its compatible torch_xla package")
    except Exception as exc:                        # TPU runtime absent or broken
        line(WARN, "PyTorch/XLA TPU", f"unavailable: {type(exc).__name__}: {exc}",
             "select a TPU runtime and keep its torch/torch_xla versions matched")

    # an answer model, if any is reachable
    if shutil.which("ollama"):
        line(OK, "ollama", "found on PATH")
    else:
        line(WARN, "ollama", "not found",
             "https://ollama.com/download (Windows-friendly alternative to vLLM)")
    if platform.system() == "Windows":
        line(WARN, "vLLM", "not supported on Windows", "use Ollama, or run vLLM inside WSL2")

    # config files present (are we in the project folder?)
    for relative in ("config/default.yaml", "config/nq_gpu.yaml", "config/nq_tpu.yaml",
                     "scripts/download_data.py"):
        if Path(relative).exists():
            line(OK, relative)
        else:
            line(MISSING, relative, "not found",
                 "cd into the extracted trace-rag folder before running this")

    # can the config actually load?
    try:
        from trace_rag.config import Config

        Config.load("config/default.yaml") if Path("config/default.yaml").exists() else Config()
        line(OK, "configuration loads")
    except Exception as exc:                       # noqa: BLE001
        line(MISSING, "configuration loads", f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} item(s) need attention: {', '.join(failures)}")
        return 1
    print("Core pipeline is ready. Next:")
    print("  python scripts/demo_end_to_end.py")
    print("  python scripts/download_data.py --dataset nq --out data/nq --queries 500  # full corpus")
    print("  T4 users: open notebooks/trace_rag_colab_t4.ipynb in Colab")
    print("  TPU users: open notebooks/trace_rag_colab_tpu.ipynb in Colab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
