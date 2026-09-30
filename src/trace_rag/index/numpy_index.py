"""Exact cosine index on numpy.  Reference implementation and test oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


class NumpyFlatIndex:
    """Exact inner-product search over L2-normalised rows.

    Used for development, for corpora below ~200k passages, and as the oracle
    the FAISS index is tested against.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._ids: List[str] = []
        self._pos: Dict[str, int] = {}
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        new_ids, new_rows = [], []
        for id_, row in zip(ids, vectors, strict=True):
            if id_ in self._pos:                      # idempotent re-add: overwrite in place
                self._vectors[self._pos[id_]] = row
                continue
            new_ids.append(id_)
            new_rows.append(row)
        if new_ids:
            start = len(self._ids)
            self._ids.extend(new_ids)
            for offset, id_ in enumerate(new_ids):
                self._pos[id_] = start + offset
            self._vectors = np.vstack([self._vectors, np.asarray(new_rows, dtype=np.float32)])

    def search(self, queries: np.ndarray, k: int, exclude: Optional[Set[str]] = None
               ) -> List[List[Tuple[str, float]]]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.shape[1] != self.dim:
            raise ValueError(f"query dim {queries.shape[1]} != index dim {self.dim}")
        if len(self._ids) == 0 or k <= 0:
            return [[] for _ in range(queries.shape[0])]

        scores = queries @ self._vectors.T                       # (q, n)
        if exclude:
            blocked = [self._pos[i] for i in exclude if i in self._pos]
            if blocked:
                scores[:, blocked] = -np.inf
        take = min(k, scores.shape[1])
        results: List[List[Tuple[str, float]]] = []
        for row in scores:
            top = np.argpartition(-row, take - 1)[:take]
            # sort by score desc, then id asc -> stable across runs and machines
            top = sorted(top, key=lambda i: (-float(row[i]), self._ids[i]))
            results.append([(self._ids[i], float(row[i])) for i in top if np.isfinite(row[i])])
        return results

    def get_vector(self, id_: str) -> Optional[np.ndarray]:
        pos = self._pos.get(id_)
        return None if pos is None else self._vectors[pos].copy()

    def get_vectors(self, ids: Sequence[str]) -> np.ndarray:
        rows = [self._pos[i] for i in ids if i in self._pos]
        return self._vectors[rows] if rows else np.zeros((0, self.dim), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), vectors=self._vectors,
                            ids=np.array(self._ids, dtype=object), dim=self.dim)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyFlatIndex":
        data = np.load(Path(path).with_suffix(".npz"), allow_pickle=True)
        index = cls(int(data["dim"]))
        ids = [str(i) for i in data["ids"].tolist()]
        index.add(ids, data["vectors"])
        return index

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._pos
