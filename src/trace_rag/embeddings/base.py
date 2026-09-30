"""Embedder base class: L2-normalised float32 output, deterministic batching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


def l2_normalise(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, eps)).astype(np.float32)


class BaseEmbedder(ABC):
    """All embedders return float32 rows; cosine similarity == dot product."""

    name: str = "base"
    dim: int = 0
    normalize: bool = True

    @abstractmethod
    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._finish(self._encode(list(texts), batch_size))

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._finish(self._encode(list(texts), batch_size))

    def _finish(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"embedder returned shape {matrix.shape}, expected 2-D")
        if matrix.shape[1] != self.dim:
            raise ValueError(f"embedder returned dim {matrix.shape[1]}, expected {self.dim}")
        if not np.isfinite(matrix).all():
            raise ValueError("embedder produced NaN/Inf")
        return l2_normalise(matrix) if self.normalize else matrix
