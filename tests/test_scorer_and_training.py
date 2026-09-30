from __future__ import annotations

import numpy as np
import pytest

from trace_rag.config import ScorerConfig
from trace_rag.contracts import Band, FeatureSnapshot, FEATURE_NAMES, SignalVector
from trace_rag.detection import (HeuristicScorer, LabelledRow, LeakageError, SuspicionScorer,
                                 TrainingSet, evaluate, leave_one_attack_out, split_by_question,
                                 train_scorer)


def synthetic_dataset(n_questions: int = 60, seed: int = 0) -> TrainingSet:
    """Clean and poisoned rows with overlapping, noisy features."""
    rng = np.random.default_rng(seed)
    dataset = TrainingSet()
    families = ["poisonedrag_bb", "corruption", "hit_and_run"]
    for q in range(n_questions):
        family = families[q % len(families)]
        for _ in range(4):
            values = np.clip(rng.normal(0.25, 0.18, size=len(FEATURE_NAMES)), 0, 1)
            dataset.add(_row(f"q{q}", "clean", 0, values, "none"))
        for _ in range(2):
            values = np.clip(rng.normal(0.62, 0.18, size=len(FEATURE_NAMES)), 0, 1)
            dataset.add(_row(f"q{q}", "poison", 1, values, family))
    return dataset


def _row(query_id: str, tag: str, label: int, values: np.ndarray, family: str) -> LabelledRow:
    signals = SignalVector(*values[:6])
    extras = {name: float(v) for name, v in zip(FEATURE_NAMES[6:], values[6:])}
    return LabelledRow(query_id=query_id, doc_id=f"{query_id}_{tag}_{label}_{values[0]:.4f}",
                       label=label, features=FeatureSnapshot(signals, extras), attack_family=family)


def test_heuristic_scorer_runs_without_training():
    scorer = HeuristicScorer()
    high = FeatureSnapshot(SignalVector(1, 1, 1, 1, 1, 1), {})
    low = FeatureSnapshot(SignalVector(0, 0, 0, 0, 0, 0), {})
    assert scorer.score_snapshot(high) == pytest.approx(1.0)
    assert scorer.score_snapshot(low) == 0.0
    assert scorer.band(0.9) is Band.HIGH and scorer.band(0.0) is Band.LOW


def test_scorer_learns_and_ranks():
    dataset = synthetic_dataset()
    scorer, report, held_out = train_scorer(dataset, ScorerConfig())
    assert held_out["roc_auc"] > 0.85
    assert 0.0 <= report.theta_low <= report.theta_high <= 1.0


def test_threshold_constraints_are_respected():
    dataset = synthetic_dataset(n_questions=90, seed=5)
    config = ScorerConfig(target_high_fpr=0.02, escalation_budget=1.0)
    scorer, report, _ = train_scorer(dataset, config)
    assert report.achieved_high_fpr <= config.target_high_fpr + 1e-9
    assert report.achieved_escalations_per_query <= config.escalation_budget + 1e-9


def test_tighter_budget_gives_higher_theta_low():
    dataset = synthetic_dataset(n_questions=90, seed=5)
    loose, _, _ = train_scorer(dataset, ScorerConfig(escalation_budget=2.0))
    tight, _, _ = train_scorer(dataset, ScorerConfig(escalation_budget=0.2))
    assert tight.theta_low >= loose.theta_low


def test_calibration_produces_probabilities():
    dataset = synthetic_dataset()
    scorer, _, _ = train_scorer(dataset, ScorerConfig(calibration="isotonic"))
    scores = scorer.predict_proba(dataset.X)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # calibrated score should correlate with the label
    assert scores[dataset.y == 1].mean() > scores[dataset.y == 0].mean()


def test_scorer_roundtrip(tmp_path):
    dataset = synthetic_dataset()
    scorer, _, _ = train_scorer(dataset, ScorerConfig())
    path = tmp_path / "scorer.joblib"
    scorer.save(path)
    reloaded = SuspicionScorer.load(path)
    assert np.allclose(reloaded.predict_proba(dataset.X[:20]), scorer.predict_proba(dataset.X[:20]))
    assert reloaded.theta_low == scorer.theta_low and reloaded.theta_high == scorer.theta_high
    assert (tmp_path / "scorer.json").exists()          # human-readable audit copy


def test_coefficients_are_interpretable():
    dataset = synthetic_dataset()
    scorer, _, _ = train_scorer(dataset, ScorerConfig())
    coefficients = scorer.coefficients()
    assert set(coefficients) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in coefficients.values())


def test_split_by_question_never_leaks():
    query_ids = [f"q{i // 5}" for i in range(200)]
    train, val, test = split_by_question(query_ids)
    groups = [set(query_ids[i] for i in split) for split in (train, val, test)]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert len(train) + len(val) + len(test) == len(query_ids)


def test_split_requires_enough_questions():
    with pytest.raises(ValueError):
        split_by_question(["q1", "q1", "q2"])


def test_leave_one_attack_out_holds_out_each_family():
    dataset = synthetic_dataset()
    folds = list(leave_one_attack_out(dataset.families))
    assert {f[0] for f in folds} == {"poisonedrag_bb", "corruption", "hit_and_run"}
    for held_out, train_idx, test_idx in folds:
        assert held_out not in {dataset.families[i] for i in train_idx}
        assert held_out in {dataset.families[i] for i in test_idx}


def test_generalisation_to_unseen_attack_family_is_measurable():
    dataset = synthetic_dataset(n_questions=90, seed=11)
    scorer, _, _ = train_scorer(dataset, ScorerConfig())
    for held_out, train_idx, test_idx in leave_one_attack_out(dataset.families):
        fold = SuspicionScorer().fit(dataset.X[train_idx], dataset.y[train_idx])
        metrics = evaluate(fold, dataset.X[test_idx], dataset.y[test_idx],
                           [dataset.query_ids[i] for i in test_idx])
        assert "roc_auc" in metrics                 # the number itself is Person C's to report


def test_feature_snapshot_after_query_time_is_rejected():
    row = LabelledRow(query_id="q1", doc_id="d1", label=1,
                      features=FeatureSnapshot(SignalVector(), {}),
                      captured_at=200.0, query_time=100.0)
    with pytest.raises(LeakageError):
        row.validate()


def test_training_set_rejects_bad_labels():
    with pytest.raises(ValueError):
        TrainingSet().add(LabelledRow("q", "d", 2, FeatureSnapshot(SignalVector(), {})))


def test_fit_requires_both_classes():
    X = np.zeros((10, len(FEATURE_NAMES)))
    with pytest.raises(ValueError):
        SuspicionScorer().fit(X, np.zeros(10, dtype=int))


def test_predict_rejects_wrong_feature_count():
    dataset = synthetic_dataset()
    scorer, _, _ = train_scorer(dataset, ScorerConfig())
    with pytest.raises(ValueError):
        scorer.predict_proba(np.zeros((2, 3)))
