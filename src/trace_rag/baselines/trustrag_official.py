"""TrustRAG's *own* filtering code, run on this system's retrieved passages.

Source: https://github.com/HuichiZhou/TrustRAG (MIT).  Clone it and point ``repo_dir`` at the
checkout; this module loads ``defend_module.py`` from there and calls the authors'
``k_mean_filtering`` unchanged.  Nothing from that repository is copied into this one.

What is and is not included -- state this in any paper that uses it:

* **Included:** stage 1, the K-means + ROUGE-L n-gram conflict filter
  (``k_mean_filtering(embedding_topk, topk_contents, adv_text_set, n_gram=True)``).  The
  ``adv_text_set`` argument is unused by the authors' function; an empty set is passed, so no
  evaluation label reaches the defence.
* **Not included:** stage 2 (LLM self-assessment / ``conflict_query``), which produces a final
  answer rather than a context filter.  Compare against the authors' full pipeline separately.
* **Deviation:** the authors embed passages with the CLS hidden state of Contriever; here the
  embeddings come from this pipeline's embedder (normalised, mean-pooled Contriever on the GPU
  config).  Thresholds in their code (0.88 similarity, 0.25 ROUGE-L) were tuned for theirs.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

from ..contracts import PolicyDecision


def load_defend_module(repo_dir: str | Path):  # type: ignore[no-untyped-def]
    """Import ``defend_module.py`` from a TrustRAG checkout.

    The file does ``from src.utils import progress_bar``; ``src.utils`` in that repository pulls in
    BEIR, sentence-transformers and an LLM stack that the filter does not need, so only
    ``progress_bar`` is provided here (a plain iterator).  Everything else is the authors' code.
    """
    path = Path(repo_dir) / "defend_module.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: git clone https://github.com/HuichiZhou/TrustRAG.git")
    saved = {name: sys.modules.get(name) for name in ("src", "src.utils")}
    stub = types.ModuleType("src.utils")
    stub.progress_bar = lambda iterable, desc=None, **_: iterable          # type: ignore[attr-defined]
    package = types.ModuleType("src")
    package.__path__ = []                                                   # type: ignore[attr-defined]
    package.utils = stub                                                    # type: ignore[attr-defined]
    sys.modules["src"], sys.modules["src.utils"] = package, stub
    try:
        spec = importlib.util.spec_from_file_location("trustrag_defend_module", path)
        module = importlib.util.module_from_spec(spec)                      # type: ignore[arg-type]
        spec.loader.exec_module(module)                                     # type: ignore[union-attr]
    except ImportError as exc:
        raise ImportError(f"TrustRAG's defend_module needs torch, scikit-learn, nltk and rouge_score "
                          f"(pip install nltk rouge_score): {exc}") from exc
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


class TrustRAGOfficialStage1Policy:
    """``Policy`` that drops the passages TrustRAG's K-means filter removes."""

    name = "trustrag_official"

    def __init__(self, repo_dir: str | Path, embed: Callable[[Sequence[str]], Any]) -> None:
        self.module = load_defend_module(repo_dir)
        self.embed = embed

    def decide(self, query, query_id, documents, assessments, verifier=None, **kwargs) -> PolicyDecision:  # type: ignore[no-untyped-def]
        docs = list(documents)
        notes: Dict[str, Any] = {"policy": self.name, "newly_quarantined": [], "stage2_included": False}
        if len(docs) < 2:
            keep_ids = [d.doc_id for d in docs]
        else:
            texts = [d.text for d in docs]
            embeddings = np.asarray(self.embed(texts), dtype=np.float64)
            _, kept_texts = self.module.k_mean_filtering(embeddings, texts, set(), True)
            budget = Counter(kept_texts)                  # multiset match: duplicate texts stay distinct
            keep_ids = []
            for doc in docs:
                if budget[doc.text] > 0:
                    budget[doc.text] -= 1
                    keep_ids.append(doc.doc_id)
        keep = set(keep_ids)
        notes["n_excluded"] = len(docs) - len(keep)
        return PolicyDecision(context_doc_ids=tuple(d.doc_id for d in docs if d.doc_id in keep),
                              excluded_doc_ids=tuple(d.doc_id for d in docs if d.doc_id not in keep),
                              verified=(), notes=notes)
