"""Public Person C evaluation API."""

from .chronological import run_stream
from .integration import to_pipeline_events, to_pipeline_queries
from .metrics import (
    RetrievalMetrics,
    retrieval_metrics,
    attack_success_rate,
    false_positive_rate,
    detection_delay,
    exposure_window,
    remediation_recall,
    llm_cost,
)
from .runner import (
    EvaluationCase,
    EvaluationResult,
    build_experiment_matrix,
    run_experiment_matrix,
    run_experiments,
)
from .statistics import bootstrap_ci, mcnemar_exact

__all__ = [
    "RetrievalMetrics", "retrieval_metrics", "attack_success_rate",
    "false_positive_rate", "detection_delay", "exposure_window",
    "remediation_recall", "llm_cost", "run_stream", "to_pipeline_queries",
    "to_pipeline_events", "EvaluationCase", "EvaluationResult",
    "build_experiment_matrix", "run_experiment_matrix", "run_experiments",
    "bootstrap_ci", "mcnemar_exact",
]
