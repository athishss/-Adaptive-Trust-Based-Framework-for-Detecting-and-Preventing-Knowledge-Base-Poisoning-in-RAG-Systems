"""Calibrated suspicion scorer and constrained band-threshold selection.

Two things here are contributions, not plumbing:

1. **Calibration.**  A raw logistic-regression score is not a probability.  We
   fit isotonic calibration on a held-out validation split so that the band
   thresholds mean what they say.

2. **Constrained thresholds.**  ``select_thresholds`` picks (theta_low,
   theta_high) to satisfy *two* operational constraints at once:
       - false-positive rate of clean passages landing in HIGH <= target,
       - mean MEDIUM-band escalations per query <= verification budget.
   Those constraints come from the plan (Section 4.4); solving them explicitly
   is what keeps the latency and LLM-cost story honest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import Band, FEATURE_NAMES, FeatureSnapshot, SignalVector


@dataclass(frozen=True)
class ThresholdReport:
    theta_low: float
    theta_high: float
    achieved_high_fpr: float
    achieved_escalations_per_query: float
    high_band_recall: float
    medium_or_high_recall: float
    n_queries: int
    n_clean: int
    n_poison: int

    def to_dict(self) -> Dict[str, float]:
        return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in asdict(self).items()}


class SuspicionScorer:
    """Logistic regression + isotonic calibration over the 13 features.

    Coefficients stay interpretable, which matters for the review: you can show
    the panel exactly which signal drove a decision.
    """

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES, C: float = 1.0,
                 calibration: str = "isotonic", random_state: int = 20260921,
                 class_weight: Optional[str] = "balanced") -> None:
        self.feature_names: Tuple[str, ...] = tuple(feature_names)
        self.C = float(C)
        self.calibration = calibration
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self.version = "untrained"
        self._pipeline = None
        self._calibrator = None
        self.theta_low = 0.30
        self.theta_high = 0.75
        self.threshold_report: Optional[ThresholdReport] = None

    # --------------------------------------------------------------- fitting
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuspicionScorer":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self._validate(X, y)
        if len(np.unique(y)) < 2:
            raise ValueError("training data must contain both clean (0) and poison (1) rows")
        self._pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, class_weight=self.class_weight,
                                      random_state=self.random_state)),
        ])
        self._pipeline.fit(X, y)
        self._calibrator = None
        self.version = f"lr-{len(self.feature_names)}f-n{X.shape[0]}"
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "SuspicionScorer":
        """Fit the probability calibrator on a *validation* split only."""
        if self._pipeline is None:
            raise RuntimeError("call fit() before calibrate()")
        if self.calibration == "none":
            return self
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val).astype(int)
        self._validate(X_val, y_val)
        raw = self._raw_scores(X_val)
        if self.calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(raw, y_val)
        elif self.calibration == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(raw.reshape(-1, 1), y_val)
        else:
            raise ValueError(f"unknown calibration: {self.calibration}")
        self._calibrator = calibrator
        self.version = f"{self.version}-{self.calibration}"
        return self

    # -------------------------------------------------------------- scoring
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("scorer is not fitted")
        return np.asarray(self._pipeline.predict_proba(np.asarray(X, dtype=np.float64))[:, 1],
                          dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {X.shape[1]}")
        raw = self._raw_scores(X)
        if self._calibrator is None:
            return raw
        if self.calibration == "isotonic":
            return np.clip(self._calibrator.predict(raw), 0.0, 1.0)
        return np.clip(self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], 0.0, 1.0)

    def score_snapshot(self, snapshot: FeatureSnapshot) -> float:
        return float(self.predict_proba(snapshot.as_array(self.feature_names)[None, :])[0])

    def band(self, suspicion: float) -> Band:
        if suspicion >= self.theta_high:
            return Band.HIGH
        if suspicion >= self.theta_low:
            return Band.MEDIUM
        return Band.LOW

    def coefficients(self) -> Dict[str, float]:
        if self._pipeline is None:
            return {}
        scale = self._pipeline.named_steps["scale"]
        coef = self._pipeline.named_steps["lr"].coef_[0]
        return {name: float(c / (s if s else 1.0))
                for name, c, s in zip(self.feature_names, coef, scale.scale_)}

    # ------------------------------------------------------------ thresholds
    def select_thresholds(self, scores: Sequence[float], labels: Sequence[int],
                          query_ids: Sequence[str], target_high_fpr: float = 0.02,
                          escalation_budget: float = 1.0) -> ThresholdReport:
        """Pick bands on validation data subject to FPR and budget constraints.

        theta_high: the *lowest* threshold whose clean-passage false-positive
        rate is still within ``target_high_fpr`` (maximises HIGH-band recall
        under the FPR cap).
        theta_low:  the *lowest* threshold that keeps mean MEDIUM escalations
        per query within ``escalation_budget`` (maximises coverage under the
        cost cap).
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        query_ids = list(query_ids)
        if not (len(scores) == len(labels) == len(query_ids)):
            raise ValueError("scores, labels and query_ids must have the same length")
        if scores.size == 0:
            raise ValueError("no validation rows supplied")
        n_queries = max(1, len(set(query_ids)))
        clean = scores[labels == 0]
        poison = scores[labels == 1]

        candidates = np.unique(np.concatenate([scores, np.array([0.0, 1.0])]))
        theta_high = 1.0
        for threshold in candidates:                       # ascending
            fpr = float((clean >= threshold).mean()) if clean.size else 0.0
            if fpr <= target_high_fpr:
                theta_high = float(threshold)
                break

        theta_low = theta_high
        for threshold in candidates:
            if threshold > theta_high:
                break
            escalations = float(((scores >= threshold) & (scores < theta_high)).sum()) / n_queries
            if escalations <= escalation_budget:
                theta_low = float(threshold)
                break

        self.theta_low, self.theta_high = float(min(theta_low, theta_high)), float(theta_high)
        report = ThresholdReport(
            theta_low=self.theta_low,
            theta_high=self.theta_high,
            achieved_high_fpr=float((clean >= self.theta_high).mean()) if clean.size else 0.0,
            achieved_escalations_per_query=float(
                ((scores >= self.theta_low) & (scores < self.theta_high)).sum()) / n_queries,
            high_band_recall=float((poison >= self.theta_high).mean()) if poison.size else 0.0,
            medium_or_high_recall=float((poison >= self.theta_low).mean()) if poison.size else 0.0,
            n_queries=n_queries, n_clean=int(clean.size), n_poison=int(poison.size),
        )
        self.threshold_report = report
        return report

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_names": list(self.feature_names),
            "pipeline": self._pipeline,
            "calibrator": self._calibrator,
            "calibration": self.calibration,
            "theta_low": self.theta_low,
            "theta_high": self.theta_high,
            "version": self.version,
            "threshold_report": self.threshold_report.to_dict() if self.threshold_report else None,
            "C": self.C,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
        }, path)
        path.with_suffix(".json").write_text(json.dumps({
            "version": self.version, "theta_low": self.theta_low, "theta_high": self.theta_high,
            "features": list(self.feature_names), "coefficients": self.coefficients(),
            "threshold_report": self.threshold_report.to_dict() if self.threshold_report else None,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SuspicionScorer":
        import joblib

        blob = joblib.load(Path(path))
        scorer = cls(feature_names=blob["feature_names"], C=blob.get("C", 1.0),
                     calibration=blob.get("calibration", "isotonic"),
                     random_state=blob.get("random_state", 20260921),
                     class_weight=blob.get("class_weight", "balanced"))
        scorer._pipeline = blob["pipeline"]
        scorer._calibrator = blob["calibrator"]
        scorer.theta_low = float(blob["theta_low"])
        scorer.theta_high = float(blob["theta_high"])
        scorer.version = blob["version"]
        report = blob.get("threshold_report")
        scorer.threshold_report = ThresholdReport(**report) if report else None
        return scorer

    def _validate(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"X must be (n, {len(self.feature_names)}), got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/Inf")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("labels must be 0/1")


class HeuristicScorer:
    """Fallback used before the scorer is trained (fresh repo, first demo).

    A fixed weighted sum of the six signals.  It is deliberately simple and is
    never used for reported numbers; ``PersonAPipeline`` logs which scorer
    produced each assessment.
    """

    version = "heuristic-v1"
    feature_names = FEATURE_NAMES

    def __init__(self, theta_low: float = 0.30, theta_high: float = 0.75,
                 weights: Optional[Dict[str, float]] = None) -> None:
        self.theta_low = float(theta_low)
        self.theta_high = float(theta_high)
        self.weights = weights or {
            "s1_query_echo": 0.30, "s2_similarity_outlier": 0.20, "s3_cluster_tightness": 0.20,
            "s4_ingestion_burst": 0.10, "s5_source_immaturity": 0.10,
            "s6_neighbourhood_density": 0.10,
        }
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score_snapshot(self, snapshot: FeatureSnapshot) -> float:
        signals = snapshot.signals
        return float(min(1.0, max(0.0, sum(
            weight * float(getattr(signals, name)) for name, weight in self.weights.items()))))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        order = {name: i for i, name in enumerate(self.feature_names)}
        return np.clip(sum(w * X[:, order[n]] for n, w in self.weights.items()), 0.0, 1.0)

    def band(self, suspicion: float) -> Band:
        if suspicion >= self.theta_high:
            return Band.HIGH
        if suspicion >= self.theta_low:
            return Band.MEDIUM
        return Band.LOW

    def coefficients(self) -> Dict[str, float]:
        return dict(self.weights)
