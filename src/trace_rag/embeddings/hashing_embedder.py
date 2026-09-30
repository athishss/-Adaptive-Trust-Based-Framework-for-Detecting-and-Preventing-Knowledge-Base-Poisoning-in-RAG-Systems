"""Dependency-free deterministic embedder.

Purpose: let the whole pipeline (and the full test suite) run without
downloading a model.  It is a hashed bag-of-n-grams projection, so it captures
lexical overlap only.  Use it for development, CI and unit tests; use
``HFEmbedder`` (Contriever / BGE) for every reported number.
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

from ..utils.textnorm import tokenise
from .base import BaseEmbedder


class HashingEmbedder(BaseEmbedder):
    name = "hashing"

    def __init__(self, dim: int = 256, ngram: int = 2, seed: int = 20260921,
                 normalize: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.ngram = max(1, int(ngram))
        self.seed = int(seed)
        self.normalize = bool(normalize)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(f"{self.seed}:{token}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _sign(self, token: str) -> float:
        digest = hashlib.blake2b(f"sign:{self.seed}:{token}".encode("utf-8"), digest_size=1).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    def _encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenise(text)
            features: List[str] = list(tokens)
            for width in range(2, self.ngram + 1):
                features.extend("_".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1))
            for feature in features:
                out[row, self._bucket(feature)] += self._sign(feature)
        return out
