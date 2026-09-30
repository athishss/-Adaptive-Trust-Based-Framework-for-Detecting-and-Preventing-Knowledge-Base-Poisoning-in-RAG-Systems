"""Training utilities for the suspicion scorer, with the leakage guards from the plan.

Three rules are enforced in code rather than left to discipline:

1. **Split by question, never by row.**  Paraphrases of one target question
   must not appear in two splits (plan Section 6.3).
2. **Thresholds and calibration come from validation only.**  The test split is
   never seen by ``train_scorer``.
3. **Features are snapshots.**  Rows carry the time they were captured; the
   builder refuses rows whose trust snapshot post-dates the query, which is how
   a future-information leak would show up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import ScorerConfig
from ..contracts import FEATURE_NAMES, FeatureSnapshot, RetrievedDocument
from .scorer import SuspicionScorer, ThresholdReport


class LeakageError(RuntimeError):
    """Raised when a split or a feature row would leak information."""


@dataclass
class LabelledRow:
    query_id: str
    doc_id: str
    label: int                       # 1 = injected/poisoned passage, 0 = clean
    features: FeatureSnapshot
    attack_family: str = "none"
    captured_at: float = 0.0
    query_time: float = 0.0

    def validate(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0/1, got {self.label}")
        if self.query_time and self.captured_at and self.captured_at > self.query_time + 1e-6:
            raise LeakageError(
                f"feature snapshot for {self.doc_id} was captured after the query "
                f"({self.captured_at} > {self.query_time}); features must be snapshotted at "
                f"retrieval time"
            )


@dataclass
class TrainingSet:
    rows: List[LabelledRow] = field(default_factory=list)
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def add(self, row: LabelledRow) -> None:
        row.validate()
        self.rows.append(row)

    def extend(self, rows: Iterable[LabelledRow]) -> None:
        for row in rows:
            self.add(row)

    @property
    def X(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack([r.features.as_array(self.feature_names) for r in self.rows])

    @property
    def y(self) -> np.ndarray:
        return np.asarray([r.label for r in self.rows], dtype=int)

    @property
    def query_ids(self) -> List[str]:
        return [r.query_id for r in self.rows]

    @property
    def families(self) -> List[str]:
        return [r.attack_family for r in self.rows]

    def subset(self, indices: Sequence[int]) -> "TrainingSet":
        return TrainingSet([self.rows[i] for i in indices], self.feature_names)

    def summary(self) -> Dict[str, object]:
        y = self.y
        return {
            "rows": len(self.rows),
            "poison": int(y.sum()) if y.size else 0,
            "clean": int((y == 0).sum()) if y.size else 0,
            "questions": len(set(self.query_ids)),
            "attack_families": sorted(set(self.families)),
        }

    def __len__(self) -> int:
        return len(self.rows)


def build_rows(query_id: str, documents: Sequence[RetrievedDocument],
               snapshots: Mapping[str, FeatureSnapshot],
               is_poison: Callable[[str], bool], attack_family: str = "none",
               query_time: float = 0.0, captured_at: Optional[float] = None) -> List[LabelledRow]:
    """Turn one retrieval into labelled rows.

    ``is_poison`` comes from Person C's evaluator-only label file.  Person A's
    pipeline never reads that file at inference time; it is passed in here, in
    the training path only.
    """
    captured = float(captured_at if captured_at is not None else query_time)
    rows: List[LabelledRow] = []
    for doc in documents:
        snapshot = snapshots.get(doc.doc_id)
        if snapshot is None:
            continue
        row = LabelledRow(
            query_id=query_id, doc_id=doc.doc_id, label=int(bool(is_poison(doc.doc_id))),
            features=snapshot, attack_family=attack_family,
            captured_at=captured, query_time=float(query_time),
        )
        row.validate()
        rows.append(row)
    return rows


def split_by_question(query_ids: Sequence[str], ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                      seed: int = 20260921) -> Tuple[List[int], List[int], List[int]]:
    """Group-wise 60/20/20 split.  No question id appears in two splits."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    questions = sorted(set(query_ids))
    if len(questions) < 3:
        raise ValueError("need at least 3 distinct questions to split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(questions))
    shuffled = [questions[i] for i in order]
    n_train = max(1, int(round(ratios[0] * len(shuffled))))
    n_val = max(1, int(round(ratios[1] * len(shuffled))))
    n_train = min(n_train, len(shuffled) - 2)
    n_val = min(n_val, len(shuffled) - n_train - 1)
    assignment = {q: "train" for q in shuffled[:n_train]}
    assignment.update({q: "val" for q in shuffled[n_train:n_train + n_val]})
    assignment.update({q: "test" for q in shuffled[n_train + n_val:]})

    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for i, qid in enumerate(query_ids):
        splits[assignment[qid]].append(i)
    assert_no_question_leakage(query_ids, splits["train"], splits["val"], splits["test"])
    return splits["train"], splits["val"], splits["test"]


def assert_no_question_leakage(query_ids: Sequence[str], *splits: Sequence[int]) -> None:
    seen: List[set] = [set(query_ids[i] for i in split) for split in splits]
    for a in range(len(seen)):
        for b in range(a + 1, len(seen)):
            overlap = seen[a] & seen[b]
            if overlap:
                raise LeakageError(f"question ids appear in two splits: {sorted(overlap)[:5]}")


def leave_one_attack_out(families: Sequence[str], query_ids: Optional[Sequence[str]] = None,
                         include_clean: str = "split", seed: int = 20260921
                         ) -> Iterator[Tuple[str, List[int], List[int]]]:
    """Yield (held_out_family, train_idx, test_idx) for generalisation testing.

    The question is "trained on attacks A and B, does it catch unseen attack C",
    so the split has to be clean in two ways:

    * no row appears on both sides - otherwise the held-out false-positive rate
      is measured on clean passages the detector was trained on;
    * no *question* appears on both sides - every row belonging to a question
      that the held-out family attacked goes to the test side, and rows from
      other attack families on those questions are dropped rather than
      contaminating either side.

    ``query_ids`` is what makes the question-level guarantee possible; without
    it the clean rows are split row-wise, which is weaker.  ``include_clean``:
    ``"split"`` (default, as described), ``"train"``/``"test"`` to force every
    clean row to one side, or ``"always"`` to reproduce the leaky behaviour for
    an ablation that deliberately wants it.
    """
    families = list(families)
    attack_families = sorted({f for f in families if f != "none"})
    if len(attack_families) < 2:
        raise ValueError("leave-one-attack-out needs at least 2 attack families")
    if include_clean not in {"split", "train", "test", "always"}:
        raise ValueError(f"unknown include_clean: {include_clean}")
    if query_ids is not None and len(query_ids) != len(families):
        raise ValueError("query_ids and families must have the same length")

    rng = np.random.default_rng(seed)
    row_split: Dict[int, bool] = {}
    if include_clean == "split" and query_ids is None:
        clean_rows = [i for i, f in enumerate(families) if f == "none"]
        chosen = set(rng.permutation(clean_rows)[: max(1, len(clean_rows) // 2)].tolist())
        row_split = {i: (i in chosen) for i in clean_rows}

    for held_out in attack_families:
        held_questions = set()
        if query_ids is not None:
            held_questions = {query_ids[i] for i, f in enumerate(families) if f == held_out}

        train_idx: List[int] = []
        test_idx: List[int] = []
        for i, family in enumerate(families):
            question = query_ids[i] if query_ids is not None else None
            if family == held_out:
                test_idx.append(i)
                continue
            if family != "none":
                if question is not None and question in held_questions:
                    continue                          # other family, held-out question: drop
                train_idx.append(i)
                continue
            # clean row
            if include_clean == "train":
                train_idx.append(i)
            elif include_clean == "test":
                test_idx.append(i)
            elif include_clean == "always":
                train_idx.append(i)
                test_idx.append(i)
            elif question is not None:
                (test_idx if question in held_questions else train_idx).append(i)
            else:
                (test_idx if row_split.get(i, False) else train_idx).append(i)
        yield held_out, train_idx, test_idx


def train_scorer(dataset: TrainingSet, config: Optional[ScorerConfig] = None,
                 ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2), seed: Optional[int] = None
                 ) -> Tuple[SuspicionScorer, ThresholdReport, Dict[str, object]]:
    """Fit -> calibrate -> choose thresholds, all without touching the test split."""
    config = config or ScorerConfig()
    seed = config.random_state if seed is None else seed
    if len(dataset) == 0:
        raise ValueError("empty training set")
    train_idx, val_idx, test_idx = split_by_question(dataset.query_ids, ratios, seed)

    X, y = dataset.X, dataset.y
    scorer = SuspicionScorer(feature_names=dataset.feature_names, C=config.C,
                             calibration=config.calibration, random_state=seed)
    scorer.fit(X[train_idx], y[train_idx])
    if config.calibration != "none":
        scorer.calibrate(X[val_idx], y[val_idx])

    val_scores = scorer.predict_proba(X[val_idx])
    report = scorer.select_thresholds(
        val_scores, y[val_idx], [dataset.query_ids[i] for i in val_idx],
        target_high_fpr=config.target_high_fpr, escalation_budget=config.escalation_budget,
    )
    held_out = evaluate(scorer, X[test_idx], y[test_idx], [dataset.query_ids[i] for i in test_idx])
    held_out["split_sizes"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    return scorer, report, held_out


def evaluate(scorer, X: np.ndarray, y: np.ndarray, query_ids: Sequence[str]) -> Dict[str, object]:
    """Ranking and operating-point metrics for a fitted scorer."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if X.shape[0] == 0:
        return {"n": 0}
    scores = scorer.predict_proba(X)
    n_queries = max(1, len(set(query_ids)))
    clean = scores[y == 0]
    poison = scores[y == 1]
    metrics: Dict[str, object] = {
        "n": int(X.shape[0]),
        "n_poison": int(poison.size),
        "n_clean": int(clean.size),
        "theta_low": float(scorer.theta_low),
        "theta_high": float(scorer.theta_high),
        "high_band_fpr": float((clean >= scorer.theta_high).mean()) if clean.size else 0.0,
        "high_band_recall": float((poison >= scorer.theta_high).mean()) if poison.size else 0.0,
        "medium_or_high_recall": float((poison >= scorer.theta_low).mean()) if poison.size else 0.0,
        "escalations_per_query": float(
            ((scores >= scorer.theta_low) & (scores < scorer.theta_high)).sum()) / n_queries,
    }
    if poison.size and clean.size:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
        metrics["average_precision"] = float(average_precision_score(y, scores))
    return metrics
