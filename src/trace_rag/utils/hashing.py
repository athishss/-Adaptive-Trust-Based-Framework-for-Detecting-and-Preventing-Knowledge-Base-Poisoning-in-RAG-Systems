"""Content hashing and near-duplicate family assignment.

The near-duplicate *family* is the middle level of the trust hierarchy
(document -> family -> source).  Two passages land in the same family when
their token shingles overlap heavily, which is how an attacker's lightly
paraphrased copies stay linked.

Implementation is a self-contained MinHash + LSH banding (no third-party
dependency, fully deterministic given the same seed).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .textnorm import shingles, tokenise

_MERSENNE = (1 << 61) - 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(item: Tuple[str, ...]) -> int:
    """Stable 64-bit hash (Python's hash() is salted per process, so not that)."""
    digest = hashlib.blake2b("\x1f".join(item).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class MinHasher:
    """Deterministic MinHash signatures over token shingles."""

    def __init__(self, num_perm: int = 64, shingle_width: int = 3, seed: int = 20260921) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = int(num_perm)
        self.shingle_width = int(shingle_width)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, _MERSENNE - 1, size=self.num_perm, dtype=np.int64)
        self._b = rng.integers(0, _MERSENNE - 1, size=self.num_perm, dtype=np.int64)

    def signature(self, text: str) -> np.ndarray:
        tokens = tokenise(text)
        grams = shingles(tokens, self.shingle_width)
        if not grams:
            return np.full(self.num_perm, _MERSENNE, dtype=np.int64)
        hashes = np.array([_stable_hash(g) % _MERSENNE for g in grams], dtype=np.int64)
        # (a * h + b) mod p, vectorised over permutations x shingles
        perm = (np.outer(self._a, hashes) + self._b[:, None]) % _MERSENNE
        return perm.min(axis=1)

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        if sig_a.shape != sig_b.shape:
            raise ValueError("signatures must have the same length")
        return float(np.mean(sig_a == sig_b))


@dataclass
class _Band:
    width: int
    buckets: Dict[bytes, str]


class FamilyAssigner:
    """Assigns a near-duplicate family id to each passage, streaming.

    ``threshold`` is the approximate Jaccard similarity at which two passages
    are considered the same family; band count is derived from it.
    """

    def __init__(self, num_perm: int = 64, bands: int = 16, shingle_width: int = 3,
                 threshold: float = 0.6, seed: int = 20260921) -> None:
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.hasher = MinHasher(num_perm=num_perm, shingle_width=shingle_width, seed=seed)
        self.bands = int(bands)
        self.rows = num_perm // bands
        self.threshold = float(threshold)
        self._band_maps: List[Dict[bytes, List[str]]] = [dict() for _ in range(self.bands)]
        self._signatures: Dict[str, np.ndarray] = {}
        self._family_of: Dict[str, str] = {}
        self._members: Dict[str, List[str]] = {}
        self._exact: Dict[str, str] = {}

    def _keys(self, signature: np.ndarray) -> List[bytes]:
        keys = []
        for band in range(self.bands):
            chunk = signature[band * self.rows:(band + 1) * self.rows]
            keys.append(hashlib.blake2b(chunk.tobytes(), digest_size=12).digest())
        return keys

    def assign(self, chunk_id: str, text: str) -> str:
        """Return the family id for ``chunk_id`` (creates one when new)."""
        signature = self.hasher.signature(text)
        exact_key = sha256_text(" ".join(tokenise(text)))
        keys = self._keys(signature)
        candidates: Set[str] = set()
        for band, key in enumerate(keys):
            candidates.update(self._band_maps[band].get(key, ()))

        family: Optional[str] = self._exact.get(exact_key)
        best = self.threshold
        for candidate in (() if family else sorted(candidates)):   # sorted -> deterministic
            score = MinHasher.jaccard(signature, self._signatures[candidate])
            if score >= best:
                best = score
                family = self._family_of[candidate]
        if family is None:
            family = f"fam_{sha256_text(chunk_id)[:16]}"

        self._exact.setdefault(exact_key, family)
        self._signatures[chunk_id] = signature
        self._family_of[chunk_id] = family
        self._members.setdefault(family, []).append(chunk_id)
        for band, key in enumerate(keys):
            self._band_maps[band].setdefault(key, []).append(chunk_id)
        return family

    def family_of(self, chunk_id: str) -> Optional[str]:
        return self._family_of.get(chunk_id)

    def family_size(self, family_id: str) -> int:
        return len(self._members.get(family_id, ()))

    def members(self, family_id: str) -> Tuple[str, ...]:
        return tuple(self._members.get(family_id, ()))

    @property
    def family_sizes(self) -> Dict[str, int]:
        return {fam: len(members) for fam, members in self._members.items()}
