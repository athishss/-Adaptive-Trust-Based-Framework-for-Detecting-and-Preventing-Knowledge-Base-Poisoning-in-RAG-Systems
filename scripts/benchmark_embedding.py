#!/usr/bin/env python3
"""Measure how fast this machine can embed passages, on CPU and on GPU.

Answers the only question that matters when indexing feels slow: is the GPU
being used, and how long will the corpus actually take?

    python scripts/benchmark_embedding.py
    python scripts/benchmark_embedding.py --model facebook/contriever --n 512

Runs in a couple of minutes, downloads nothing that is not already cached, and
ends with a recommended device, batch size and corpus size.
"""

from __future__ import annotations

import argparse
import platform
import time
from typing import Dict, List, Optional

SAMPLE = (
    "Gustave Eiffel was the French civil engineer whose company designed and built the Eiffel "
    "Tower in Paris, completed in 1889 for the World's Fair. The wrought-iron lattice tower "
    "stands on the Champ de Mars and remains one of the most visited monuments in the world."
)


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def bench(device: str, model_name: str, texts: List[str], batch_size: int) -> Optional[float]:
    from trace_rag.embeddings.hf_embedder import HFEmbedder

    try:
        embedder = HFEmbedder(model_name=model_name, device=device)
    except Exception as exc:                      # noqa: BLE001
        print(f"  {device}: unavailable ({type(exc).__name__}: {str(exc).splitlines()[0]})")
        return None
    embedder.encode_documents(texts[:batch_size], batch_size=batch_size)      # warm up
    start = time.perf_counter()
    embedder.encode_documents(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    rate = len(texts) / elapsed if elapsed else 0.0
    print(f"  {device:4s}: {rate:8.0f} passages/s   ({elapsed:.1f}s for {len(texts)})")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/contriever")
    parser.add_argument("--n", type=int, default=512, help="passages per timing run")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"TRACE-RAG embedding benchmark\npython {platform.python_version()} on "
          f"{platform.system()}\nmodel: {args.model}\n")

    try:
        import torch
    except ImportError:
        print("torch is not installed; install it before indexing with a real model")
        return 1

    print(f"torch {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}, {properties.total_memory / 1024 ** 3:.1f} GB, "
              f"{properties.multi_processor_count} SMs")
    else:
        print("GPU: none visible to torch")
    print()

    texts = [f"{SAMPLE} Passage {i}." for i in range(args.n)]
    print(f"timing {args.n} passages at batch size {args.batch_size}:")
    rates: Dict[str, Optional[float]] = {"cpu": bench("cpu", args.model, texts, args.batch_size)}
    if cuda:
        rates["cuda"] = bench("cuda", args.model, texts, args.batch_size)

    best_device = max((d for d, r in rates.items() if r), key=lambda d: rates[d] or 0.0,
                      default="cpu")
    best_rate = rates.get(best_device) or 1.0
    print(f"\nfastest device: {best_device} at {best_rate:.0f} passages/s")
    print("projected indexing time:")
    for size, label in ((20_000, "20k subset"), (200_000, "200k subset"), (2_681_468, "full NQ")):
        print(f"  {label:<12} {human_time(size / best_rate)}")

    print()
    if cuda and rates.get("cuda") and rates.get("cpu") and rates["cuda"] < rates["cpu"] * 2:
        print("WARNING: the GPU is barely faster than the CPU. Check that nothing else is using "
              "it (nvidia-smi), and try --batch-size 128.")
    if best_rate < 100:
        print("This machine embeds slowly. Options, in order of effort:")
        print("  1. index a smaller corpus:   --limit 20000   (enough to validate the pipeline)")
        print("  2. raise the batch size:     --set embedding.batch_size=128")
        print("  3. run the indexing step on a machine with a stronger GPU, copy runs/ back")
    else:
        print(f"Recommended: --set embedding.device={best_device} "
              f"--set embedding.batch_size={args.batch_size * 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
