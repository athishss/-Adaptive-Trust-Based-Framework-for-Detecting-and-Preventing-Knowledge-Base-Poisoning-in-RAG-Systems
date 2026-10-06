"""The six cheap always-on signals (plan Section 4.3).

Every signal is in [0, 1]; higher means more suspicious.  None of them calls an
LLM, and all of them are computed from data available *at retrieval time*.

Provenance of the ideas (stated honestly in the report):
  S1, S3 come from published observations (PoisonedRAG builds passages as
  query + injected text; TrustRAG observes that injected passages cluster).
  S2, S4, S5, S6 are ours; the contribution is the combination with trust and
  source history plus the leakage-guarded training in ``detection.training``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SignalsConfig
from ..contracts import EXTRA_FEATURE_NAMES, FeatureSnapshot, RetrievedDocument, SignalVector
from ..ingestion.provenance import ProvenanceStore
from ..utils.textnorm import tokenise


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _logistic(x: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-x / scale)) if abs(x / scale) < 60 else (0.0 if x < 0 else 1.0)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length (rolling row, O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def query_echo(query: str, text: str, prefix_tokens: int = 128) -> float:
    """S1: how much of the query is echoed verbatim near the start of the passage.

    PoisonedRAG's black-box construction is P = Q (+) I, so the question appears
    literally inside the passage.  Three views are combined and the strongest
    wins:

      * bigram containment - catches verbatim echo, ignores topic-word overlap;
      * LCS recall         - catches the query with a few words changed;
      * unigram containment, damped - weak evidence only, because any on-topic
        passage naturally shares the query's content words.

    A relevant clean passage scores well below a passage that restates the
    question, which is what stops S1 from simply firing on relevance.
    """
    q_tokens = tokenise(query)
    if not q_tokens:
        return 0.0
    d_tokens = tokenise(text)[:prefix_tokens]
    if not d_tokens:
        return 0.0

    unigram = len(set(q_tokens) & set(d_tokens)) / len(set(q_tokens))
    lcs_recall = lcs_length(q_tokens, d_tokens) / len(q_tokens)
    if len(q_tokens) >= 2:
        q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
        d_bigrams = {(d_tokens[i], d_tokens[i + 1]) for i in range(len(d_tokens) - 1)}
        bigram = len(q_bigrams & d_bigrams) / len(q_bigrams)
    else:
        bigram = unigram
    return _clip01(max(bigram, lcs_recall, 0.5 * unigram))


def similarity_outlier(similarity: float, pool_similarities: Sequence[float],
                       local_window: int = 20, min_gap_ratio: float = 3.0,
                       saturation_ratio: float = 18.0) -> float:
    """S2: is this passage part of a cluster detached from the similarity curve?

    Retrieval similarities decay smoothly: each rank sits slightly below the one
    above it.  Passages optimised for the query do not join that curve, they sit
    above it with a visible gap underneath - and because an attacker injects
    several, the gap appears below the whole injected group.

    So the score is driven by the largest gap in the top of the pool, measured
    against the typical gap there.  Everything above an unusually large gap is
    flagged; everything below it scores zero.

    Comparing a passage to the pool as a whole (the obvious approach) does not
    work: signals are only computed for the top-k, which are the highest
    similarities in the pool by construction.  Measured on real Natural
    Questions retrieval that gave ordinary clean passages 0.8-0.98.
    """
    values = sorted((float(s) for s in pool_similarities if math.isfinite(s)),
                    reverse=True)[:max(6, local_window)]
    if len(values) < 6:
        return 0.0
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    if not gaps or max(gaps) <= 0:
        return 0.0
    typical = float(np.median([g for g in gaps if g >= 0])) if gaps else 0.0
    largest = max(gaps)
    boundary = gaps.index(largest)                 # passages at ranks <= boundary sit above it
    if typical <= 1e-9:
        typical = 1e-9
    ratio = largest / typical
    if ratio <= min_gap_ratio:
        return 0.0

    position = next((i for i, v in enumerate(values)
                     if abs(v - float(similarity)) < 1e-12), len(values))
    if position > boundary:
        return 0.0
    span = max(1e-9, saturation_ratio - min_gap_ratio)
    return _clip01((ratio - min_gap_ratio) / span)


def cluster_tightness(target_vector: np.ndarray, pool_vectors: np.ndarray,
                      pool_source_new: Sequence[float], self_index: int,
                      similarity_threshold: float = 0.8) -> float:
    """S3: tight mutual similarity with other passages, weighted by source newness.

    Injected passages for one target question tend to agree with each other far
    more than independent evidence does.  Agreement only counts as suspicious
    when it comes from *new* sources: a cluster of long-standing passages is
    just a well-covered fact.
    """
    if pool_vectors.shape[0] <= 1:
        return 0.0
    similarities = pool_vectors @ np.asarray(target_vector, dtype=np.float32)
    mask = np.ones(similarities.shape[0], dtype=bool)
    if 0 <= self_index < mask.size:
        mask[self_index] = False
    similarities = similarities[mask]
    newness = np.asarray(pool_source_new, dtype=np.float64)[mask]
    if similarities.size == 0:
        return 0.0
    close = similarities >= similarity_threshold
    if not close.any():
        return 0.0
    density = float(close.sum()) / float(similarities.size)
    mean_newness = float(newness[close].mean()) if newness.size else 0.0
    mean_similarity = float(similarities[close].mean())
    return _clip01(density * mean_newness * mean_similarity * 3.0)


def ingestion_burst(burst_count: int, source_total: int, window_hours: float = 24.0,
                    reference: int = 25) -> float:
    """S4: many passages from one source in a short window, relative to its history.

    Honest contributors trickle content in; poisoning campaigns arrive in
    batches.  Normalised by the source's own volume so a large, legitimately
    busy source is not permanently suspicious.
    """
    burst_count = max(0, int(burst_count))
    source_total = max(1, int(source_total))
    if burst_count <= 1:
        return 0.0
    magnitude = math.log1p(burst_count) / math.log1p(max(reference, 2))
    concentration = burst_count / source_total          # 1.0 = the source is only this burst
    return _clip01(magnitude * concentration)


def source_immaturity(age_days: float, n_docs: int, source_trust: float,
                      mature_days: float = 180.0, mature_docs: int = 50) -> float:
    """S5: how little history stands behind this passage's source.

    Deliberately *not* "new means bad": it saturates at zero once a source has
    either enough age or enough verified content, and it is damped by the
    source's trust so an established-but-new-ish contributor is not punished.
    """
    age_score = 1.0 - _clip01(float(age_days) / max(1e-6, float(mature_days)))
    volume_score = 1.0 - _clip01(float(n_docs) / max(1, int(mature_docs)))
    immaturity = 0.5 * age_score + 0.5 * volume_score
    trust_damping = 0.5 + 0.5 * (1.0 - _clip01(float(source_trust)))
    return _clip01(immaturity * trust_damping)


def neighbourhood_density(target_vector: np.ndarray, index, k: int = 10,  # type: ignore[no-untyped-def]
                          exclude_ids: Optional[Sequence[str]] = None,
                          duplicate_threshold: float = 0.95,
                          min_corpus_for_isolation: int = 10_000) -> float:
    """S6: how the passage sits in its corpus neighbourhood.

    Two effects are measured:

    * **duplication** - a swarm of near-identical neighbours, which is what a
      batch of injected passages looks like;
    * **isolation** - the passage sits far from anything else in the corpus.

    Isolation is only used once the corpus is large enough for a neighbourhood
    to mean something (``min_corpus_for_isolation``).  On a small corpus every
    genuinely unique fact looks isolated, so using it there would punish clean
    content - a false-positive source we would rather not have.
    """
    vector = np.asarray(target_vector, dtype=np.float32)
    exclude = set(exclude_ids or ())
    hits = index.search(vector[None, :], k + len(exclude) + 1, exclude=None)[0]
    scores = [s for id_, s in hits if id_ not in exclude][:k + 1]
    if len(scores) < 3:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    neighbours = arr[1:]                                  # arr[0] is the passage itself
    duplication = float((neighbours >= duplicate_threshold).mean())
    if len(index) < min_corpus_for_isolation:
        return _clip01(duplication)
    gap = float(arr[0]) - float(neighbours.mean())
    return _clip01(max(duplication, _clip01(gap * 2.0)))


@dataclass
class _SourceInfo:
    age_days: float
    n_docs: int
    n_chunks: int
    burst: int


class SignalComputer:
    """Computes S1..S6 plus the raw extras for a retrieval outcome.

    All look-ups are cached per call, so the cost is one index search per
    retrieved passage (S6) and a handful of SQLite reads, with no LLM calls.
    """

    def __init__(self, store: ProvenanceStore, index, config: Optional[SignalsConfig] = None
                 ) -> None:  # type: ignore[no-untyped-def]
        self.store = store
        self.index = index
        self.config = config or SignalsConfig()

    def _source_info(self, source_id: str, ingested_at: float, now: float) -> _SourceInfo:
        stats = self.store.source_stats(source_id, as_of=now)
        if stats is None:
            return _SourceInfo(age_days=0.0, n_docs=0, n_chunks=0, burst=1)
        burst = self.store.burst_count(source_id, ingested_at,
                                       self.config.burst_window_hours, as_of=now)
        return _SourceInfo(age_days=stats.age_days(now), n_docs=stats.n_docs,
                           n_chunks=stats.n_chunks, burst=burst)

    def compute(self, query: str, documents: Sequence[RetrievedDocument],
                pool: Sequence[RetrievedDocument], now: Optional[float] = None
                ) -> Dict[str, FeatureSnapshot]:
        """Return feature snapshots keyed by doc_id, for the top-k ``documents``."""
        now = time.time() if now is None else float(now)
        enabled = self.config.enabled
        pool = list(pool) or list(documents)
        pool_ids = [d.doc_id for d in pool]
        pool_similarities = [d.similarity for d in pool]

        vectors = []
        for doc in pool:
            vector = self.index.get_vector(doc.doc_id)
            vectors.append(np.zeros(self.index.dim, dtype=np.float32) if vector is None else vector)
        pool_vectors = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), np.float32)

        source_cache: Dict[Tuple[str, float], _SourceInfo] = {}
        newness: List[float] = []
        for doc in pool:
            ingested = float(doc.metadata.get("ingested_at", now))
            key = (doc.source_id, round(ingested, 3))
            if key not in source_cache:
                source_cache[key] = self._source_info(doc.source_id, ingested, now)
            info = source_cache[key]
            newness.append(1.0 - _clip01(info.age_days / max(1e-6, self.config.source_mature_days)))

        family_sizes = self.store.family_sizes(
            [d.family_id for d in documents], as_of=now,
        )
        index_by_id = {doc_id: i for i, doc_id in enumerate(pool_ids)}

        out: Dict[str, FeatureSnapshot] = {}
        for doc in documents:
            ingested = float(doc.metadata.get("ingested_at", now))
            info = source_cache.get((doc.source_id, round(ingested, 3))) or \
                self._source_info(doc.source_id, ingested, now)
            position = index_by_id.get(doc.doc_id, -1)
            target_vector = (pool_vectors[position] if 0 <= position < pool_vectors.shape[0]
                             else self.index.get_vector(doc.doc_id))
            if target_vector is None:
                target_vector = np.zeros(pool_vectors.shape[1] if pool_vectors.size else 1, np.float32)

            s1 = query_echo(query, doc.text) if enabled.get("s1", True) else 0.0
            s2 = (similarity_outlier(doc.similarity, pool_similarities)
                  if enabled.get("s2", True) else 0.0)
            s3 = (cluster_tightness(target_vector, pool_vectors, newness, position,
                                    self.config.cluster_similarity_threshold)
                  if enabled.get("s3", True) and pool_vectors.size else 0.0)
            s4 = (ingestion_burst(info.burst, info.n_chunks, self.config.burst_window_hours)
                  if enabled.get("s4", True) else 0.0)
            s5 = (source_immaturity(info.age_days, info.n_docs, doc.trust.t_source,
                                    self.config.source_mature_days, self.config.source_mature_docs)
                  if enabled.get("s5", True) else 0.0)
            s6 = (neighbourhood_density(target_vector, self.index, self.config.neighbourhood_k,
                                        duplicate_threshold=self.config.duplicate_threshold,
                                        min_corpus_for_isolation=self.config.min_corpus_for_isolation)
                  if enabled.get("s6", True) else 0.0)

            signals = SignalVector(s1, s2, s3, s4, s5, s6)
            extras = {
                "x_similarity": float(doc.similarity),
                "x_rank": float(doc.rank),
                "x_source_log_age_days": float(math.log1p(max(0.0, info.age_days))),
                "x_source_n_docs": float(info.n_docs),
                "x_source_trust": float(doc.trust.t_source),
                "x_doc_trust": float(doc.trust.t_doc),
                "x_family_size": float(family_sizes.get(doc.family_id, 1)),
            }
            missing = set(EXTRA_FEATURE_NAMES) - set(extras)
            if missing:                                 # guards contract drift
                raise RuntimeError(f"missing extra features: {sorted(missing)}")
            out[doc.doc_id] = FeatureSnapshot(signals=signals, extras=extras)
        return out
