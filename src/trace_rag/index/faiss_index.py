"""FAISS index for the full corpora (NQ 2.68M, HotpotQA 5.23M, MS-MARCO 8.84M).

Kinds:
  * ``flat``  - exact, ~8.2 GB RAM for 2.68M x 768 float32.  Ground truth.
  * ``ivfpq`` - compressed, what you actually run the full corpora on.
  * ``hnsw``  - graph index, fast queries, larger memory than PQ.

Vectors must be L2-normalised, so inner product == cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FaissIndex:
    def __init__(self, dim: int, kind: str = "flat", nlist: int = 4096, pq_m: int = 96,
                 nbits: int = 8, hnsw_m: int = 32, nprobe: int = 16, seed: int = 20260921) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("FaissIndex needs faiss: pip install faiss-cpu (or faiss-gpu)") from exc
        self._faiss = faiss
        self.dim = int(dim)
        self.kind = kind
        self.nlist = int(nlist)
        self.pq_m = int(pq_m)
        self.nbits = int(nbits)
        self.hnsw_m = int(hnsw_m)
        self.nprobe = int(nprobe)
        self.seed = int(seed)
        self._ids: List[Optional[str]] = []      # position -> id, None marks a tombstone
        self._pos: Dict[str, int] = {}           # id -> current position
        self._dead = 0
        self._index = self._build()

    def _build(self):  # type: ignore[no-untyped-def]
        faiss = self._faiss
        if self.kind == "flat":
            return faiss.IndexFlatIP(self.dim)
        if self.kind == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            return index
        if self.kind == "ivfpq":
            if self.dim % self.pq_m != 0:
                raise ValueError(f"pq_m ({self.pq_m}) must divide dim ({self.dim})")
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.pq_m, self.nbits,
                                     faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe
            return index
        raise ValueError(f"unknown faiss kind: {self.kind}")

    @property
    def is_trained(self) -> bool:
        return bool(self._index.is_trained)

    def train(self, vectors: np.ndarray) -> None:
        """IVF-PQ needs training on a representative sample before adding."""
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if not self._index.is_trained:
            n = int(vectors.shape[0])
            if n < self.nlist:
                raise ValueError(
                    f"cannot train an ivfpq index with nlist={self.nlist} on {n} vectors; "
                    f"lower nlist in the config (it must not exceed the corpus size)"
                )
            recommended = 39 * max(self.nlist, 2 ** self.nbits)
            if n < recommended:
                logger.warning(
                    "training ivfpq on %d vectors; about %d are recommended for nlist=%d / "
                    "nbits=%d, so recall will suffer", n, recommended, self.nlist, self.nbits)
            self._index.train(vectors)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        if not self._index.is_trained:
            self.train(vectors)
        rows = np.ascontiguousarray(np.asarray(list(vectors), dtype=np.float32))
        start = len(self._ids)
        self._index.add(rows)
        for offset, id_ in enumerate(ids):
            previous = self._pos.get(id_)
            if previous is not None:
                # FAISS cannot update a row in place (and HNSW cannot remove one),
                # so the old position is tombstoned and searches skip it.  The new
                # vector wins; rebuild the index if tombstones pile up.
                self._ids[previous] = None
                self._dead += 1
            self._ids.append(id_)
            self._pos[id_] = start + offset

    def search(self, queries: np.ndarray, k: int, exclude: Optional[Set[str]] = None
               ) -> List[List[Tuple[str, float]]]:
        queries = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        if queries.ndim == 1:
            queries = queries[None, :]
        if len(self._ids) == 0 or k <= 0:
            return [[] for _ in range(queries.shape[0])]
        # over-fetch so that excluded ids can be dropped without losing depth
        fetch = min(len(self._ids), k + self._dead + (len(exclude) if exclude else 0))
        scores, indices = self._index.search(queries, fetch)
        results: List[List[Tuple[str, float]]] = []
        for row_scores, row_idx in zip(scores, indices):
            hits: List[Tuple[str, float]] = []
            for score, idx in zip(row_scores, row_idx):
                if idx < 0:
                    continue
                id_ = self._ids[idx]
                if id_ is None:                      # tombstoned by a later re-add
                    continue
                if exclude and id_ in exclude:
                    continue
                hits.append((id_, float(score)))
                if len(hits) == k:
                    break
            hits.sort(key=lambda pair: (-pair[1], pair[0]))
            results.append(hits)
        return results

    def get_vector(self, id_: str) -> Optional[np.ndarray]:
        pos = self._pos.get(id_)
        if pos is None:
            return None
        try:
            return np.asarray(self._index.reconstruct(int(pos)), dtype=np.float32)
        except RuntimeError:                       # pragma: no cover - kind without reconstruct
            return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        meta = {"dim": self.dim, "kind": self.kind, "nlist": self.nlist, "pq_m": self.pq_m,
                "nbits": self.nbits, "hnsw_m": self.hnsw_m, "nprobe": self.nprobe,
                "ids": self._ids, "dead": self._dead}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        import faiss  # type: ignore

        from .numpy_index import IndexLoadError

        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise IndexLoadError(
                f"cannot read the index metadata at {meta_path}: {type(exc).__name__}: {exc}. "
                f"Rebuild the index with: trace-rag index --rebuild"
            ) from exc
        index = cls(dim=meta["dim"], kind=meta["kind"], nlist=meta["nlist"], pq_m=meta["pq_m"],
                    nbits=meta["nbits"], hnsw_m=meta["hnsw_m"], nprobe=meta["nprobe"])
        try:
            index._index = faiss.read_index(str(path.with_suffix(".faiss")))
        except Exception as exc:
            raise IndexLoadError(
                f"cannot read the FAISS index at {path.with_suffix('.faiss')}: "
                f"{type(exc).__name__}: {exc}. Rebuild it with: trace-rag index --rebuild"
            ) from exc
        if meta["kind"] == "ivfpq":
            index._index.nprobe = meta["nprobe"]
        index._ids = list(meta["ids"])
        index._dead = int(meta.get("dead", 0))
        index._pos = {id_: i for i, id_ in enumerate(index._ids) if id_ is not None}
        return index

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(id_ for id_ in self._ids if id_ is not None)

    @property
    def dead_rows(self) -> int:
        """Tombstoned rows left behind by re-adding an existing id."""
        return self._dead

    def __len__(self) -> int:
        return len(self._pos)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._pos
