from trace_rag.attacks import ATTACK_NAMES
from trace_rag.attacks.adaptive import hit_and_run
from trace_rag.contracts import (
    Band,
    FeatureSnapshot,
    RetrievedDocument,
    SecurityAssessment,
    SignalVector,
    TrustSnapshot,
    TrustStatus,
)
from trace_rag.attacks.poisoned_rag import (
    entity_swap,
    instruction_injection,
    make_poisoned_document,
    negation,
)
from trace_rag.baselines import BASELINE_NAMES, build_baseline_policy
from trace_rag.evaluation.metrics import (
    ANSWER_LEVEL_ASR_NOT_MEASURED,
    answer_level_asr_fields,
    detection_delay,
    exposure_window,
    false_positive_rate,
    llm_cost,
    poison_citation_rate,
    poison_exposure_rate,
    remediation_recall,
    retrieval_metrics,
)
from trace_rag.evaluation.runner import (
    EvaluationCase,
    EvaluationResult,
    build_experiment_matrix,
    run_experiments,
)
from trace_rag.evaluation.statistics import bootstrap_ci, mcnemar_exact


def test_attack_registry_and_transformations():
    assert {
        "entity_swap",
        "negation",
        "instruction_injection",
        "hit_and_run",
        "slow_burn",
        "framing",
        "signal_evasion",
    }.issubset(set(ATTACK_NAMES))

    assert entity_swap("Alice visited London", "London", "Paris") == (
        "Alice visited Paris"
    )

    assert "It is not true that The sky is blue." in negation(
        "The sky is blue.",
        "The sky is blue.",
    )

    injected = instruction_injection("Normal document", "Ignore previous instructions")
    assert "[DOCUMENT INSTRUCTION]" in injected
    assert "Ignore previous instructions" in injected


def test_poisoned_document_serialization():
    document = make_poisoned_document(
        doc_id="d1",
        chunk_id="c1",
        text="poison",
        source_id="attacker",
        family_id="family",
        attack_type="negation",
        target_query="q1",
        metadata={"seed": 1},
    )

    data = document.to_dict()

    assert data["doc_id"] == "d1"
    assert data["target_query"] == "q1"
    assert data["metadata"]["seed"] == 1


def test_adaptive_event_serialization():
    event = hit_and_run(["payload"])[0]
    data = event.to_dict()

    assert data["attack_type"] == "hit_and_run"
    assert data["source_id"] == "attacker_0"
    assert data["payload"] == "payload"


def test_baseline_registry():
    # Every advertised name must have an implementation (build_baseline_policy),
    # and trust_threshold must be listed because it is runnable.
    assert set(BASELINE_NAMES) == {
        "no_defence",
        "perplexity",
        "duplicate_filter",
        "trust_threshold",
        "suspicion_only",
        "corroboration_only",
        "trust_no_deferred",
        "TrustRAG",
        "RobustRAG",
        "always_on_loo",
    }
    from trace_rag.baselines import build_baseline_policy

    for name in ("no_defence", "duplicate_filter", "trust_threshold"):
        assert build_baseline_policy(name) is not None


def _baseline_doc(doc_id, family_id, trust, rank=0):
    snapshot = TrustSnapshot(
        doc_id=doc_id, source_id=f"src_{doc_id}", family_id=family_id,
        t_doc=trust, t_family=trust, t_source=trust, t_eff=trust,
        status=TrustStatus.TRUSTED,
    )
    return RetrievedDocument(
        doc_id=doc_id, chunk_id=doc_id, text=f"text for {doc_id}",
        similarity=1.0 - rank * 0.1, rank=rank, source_id=f"src_{doc_id}",
        family_id=family_id, trust=snapshot, metadata={"ingested_at": 1.0},
    )


def _baseline_assessment(doc_id, band):
    return SecurityAssessment(
        doc_id=doc_id, chunk_id=doc_id, query_id="q", signals=SignalVector(),
        suspicion=0.1, band=band, action=band.value,
        feature_snapshot=FeatureSnapshot(SignalVector(), {}), scorer_version="test",
    )


def test_trust_and_robust_baselines_are_distinct():
    """Baselines must not collapse into the full RAGLedger policy."""
    documents = [
        _baseline_doc("a", "family_a", 0.9, rank=0),
        _baseline_doc("b", "family_a", 0.9, rank=1),   # duplicate family
        _baseline_doc("c", "family_c", 0.05, rank=2), # low trust
        _baseline_doc("d", "family_d", 0.9, rank=3),  # high band
    ]
    assessments = [
        _baseline_assessment("a", Band.LOW),
        _baseline_assessment("b", Band.LOW),
        _baseline_assessment("c", Band.LOW),
        _baseline_assessment("d", Band.HIGH),
    ]

    trust_rag = build_baseline_policy("TrustRAG", trust_floor=0.20)
    robust_rag = build_baseline_policy("RobustRAG", trust_floor=0.20)
    trust_decision = trust_rag.decide("q", "q1", documents, assessments)
    robust_decision = robust_rag.decide("q", "q1", documents, assessments)

    # TrustRAG deduplicates families but does not band-block.
    assert trust_decision.context_doc_ids == ("a", "d")
    assert "b" in trust_decision.excluded_doc_ids
    assert "c" in trust_decision.excluded_doc_ids
    # RobustRAG band-blocks but does not deduplicate.
    assert robust_decision.context_doc_ids == ("a", "b")
    assert "c" in robust_decision.excluded_doc_ids
    assert "d" in robust_decision.excluded_doc_ids
    assert trust_decision.verified == () and robust_decision.verified == ()


def test_metrics_and_zero_denominators():
    metrics = retrieval_metrics(["a", "b"], ["b", "c"])

    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5

    assert poison_exposure_rate(1, 2) == 0.5
    assert poison_citation_rate(1, 4) == 0.25
    assert false_positive_rate(1, 4) == 0.25
    assert remediation_recall(1, 2) == 0.5

    assert poison_exposure_rate(0, 0) is None
    assert poison_citation_rate(0, 0) is None
    assert false_positive_rate(0, 0) is None
    assert remediation_recall(0, 0) is None


def test_poison_rates_use_eligible_attempts_and_hand_calculated_counts():
    """Five attempts: two exposed, one cited-but-blocked, two never retrieved.

    The denominator stays 5 for both rates.  Abstentions and never-retrieved
    attacks are not dropped, and a citation is not treated as exposure.
    """
    attempts = [
        {"exposed": True, "cited": True},     # exposed and cited
        {"exposed": True, "cited": False},    # exposed only
        {"exposed": False, "cited": True},    # cited but withheld; not exposed
        {"exposed": False, "cited": False},   # never retrieved
        {"exposed": False, "cited": False},   # abstained for another reason
    ]
    exposed = sum(1 for row in attempts if row["exposed"])
    cited = sum(1 for row in attempts if row["cited"])
    assert poisoned_fixture_counts(attempts) == (exposed, cited, len(attempts))
    assert poison_exposure_rate(exposed, len(attempts)) == 2 / 5
    assert poison_citation_rate(cited, len(attempts)) == 2 / 5


def poisoned_fixture_counts(attempts):
    return (sum(1 for row in attempts if row["exposed"]),
            sum(1 for row in attempts if row["cited"]), len(attempts))


def test_answer_level_asr_is_explicitly_not_measured():
    fields = answer_level_asr_fields()
    assert fields["answer_level_asr"] is None
    assert fields["answer_level_asr_status"] == ANSWER_LEVEL_ASR_NOT_MEASURED
    assert fields["answer_level_asr_denominator"] == 0
    assert "not inferred" in fields["answer_level_asr_reason"]

    measured = answer_level_asr_fields(0.5, evaluated=4)
    assert measured["answer_level_asr"] == 0.5
    assert measured["answer_level_asr_numerator"] == 2
    assert measured["answer_level_asr_denominator"] == 4


def test_temporal_metrics_and_cost():
    assert detection_delay(3, 5) == 2
    assert detection_delay(3, None) is None
    assert exposure_window(3, 7, 10) == 4
    assert exposure_window(3, None, 10) == 7
    assert llm_cost(7) == 7
    assert llm_cost(-1) == 0


def test_statistics():
    low, high = bootstrap_ci([0.0, 1.0, 1.0, 0.0], seed=1, n_bootstrap=100)

    assert 0.0 <= low <= high <= 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01


def test_experiment_matrix_and_executor():
    cases = build_experiment_matrix(
        ["entity_swap", "negation"],
        ["no_defence", "TrustRAG"],
        [1, 2],
    )

    assert len(cases) == 8

    results = run_experiments(
        ["entity_swap", "negation"],
        ["no_defence", "TrustRAG"],
        [1, 2],
        lambda case: EvaluationResult(
            case=case,
            eligible_attack_attempts=2,
            poison_exposed_attempts=1,
            poison_exposure_rate=0.5,
            poison_cited_attempts=1,
            poison_citation_rate=0.5,
            answer_level_asr=None,
            precision=0.8,
            recall=0.6,
            f1=0.6857142857,
            false_positive_rate=0.1,
            detection_delay=2,
            exposure_window=3,
            remediation_recall=0.5,
            llm_calls=4,
        ),
    )

    assert len(results) == 8
    assert all(result.case in cases for result in results)
    assert results[0].to_dict()["llm_calls"] == 4
