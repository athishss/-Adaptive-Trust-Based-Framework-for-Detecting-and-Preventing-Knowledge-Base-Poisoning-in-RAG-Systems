from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "trace_rag") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("TRACE_RAG_LOG", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        # Third-party libraries log every HTTP request at INFO, which buries our
        # own output during a model download.  Raise their floor to WARNING
        # unless the user explicitly asked for DEBUG.
        if level != "DEBUG":
            for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock",
                          "faiss", "faiss.loader", "transformers"):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
