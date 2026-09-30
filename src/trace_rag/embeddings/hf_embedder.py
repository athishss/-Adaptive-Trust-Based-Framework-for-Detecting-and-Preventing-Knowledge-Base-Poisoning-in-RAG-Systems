"""Transformer embedder for Contriever / BGE / E5 (the models used for reported results).

Mean pooling with attention mask is what Contriever's model card specifies;
BGE uses CLS pooling and a query prefix, so pooling is selectable.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import numpy as np

from .base import BaseEmbedder


class HFEmbedder(BaseEmbedder):
    name = "huggingface"

    def __init__(self, model_name: str = "facebook/contriever", device: str = "cpu",
                 pooling: Literal["mean", "cls"] = "mean", max_length: int = 512,
                 query_prefix: str = "", document_prefix: str = "", normalize: bool = True,
                 dtype: str = "float32") -> None:
        try:
            import torch                              # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "HFEmbedder needs 'torch' and 'transformers'. Install with: "
                "pip install 'trace-rag[models]'"
            ) from exc
        import torch

        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.max_length = int(max_length)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize = bool(normalize)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def _pool(self, hidden, mask):  # type: ignore[no-untyped-def]
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _encode_with_prefix(self, texts: Sequence[str], batch_size: int, prefix: str) -> np.ndarray:
        torch = self._torch
        vectors: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [f"{prefix}{t}" for t in texts[start:start + batch_size]]
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt").to(self.device)
                output = self.model(**encoded)
                pooled = self._pool(output.last_hidden_state, encoded["attention_mask"])
                vectors.append(pooled.float().cpu().numpy())
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(vectors)

    def _encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._encode_with_prefix(texts, batch_size, self.document_prefix)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._finish(self._encode_with_prefix(list(texts), batch_size, self.document_prefix))

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._finish(self._encode_with_prefix(list(texts), batch_size, self.query_prefix))
