"""Typed configuration.  Every threshold in the plan lives here, never in code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class IngestionConfig(BaseModel):
    chunk_words: int = Field(100, ge=20, le=1024, description="~100 words matches BEIR passage size")
    chunk_overlap_words: int = Field(20, ge=0)
    min_chunk_words: int = Field(15, ge=1)
    minhash_perm: int = Field(64, ge=16)
    minhash_bands: int = Field(16, ge=1)
    shingle_width: int = Field(3, ge=1)
    family_threshold: float = Field(0.6, ge=0.0, le=1.0)
    family_backend: Literal["minhash", "exact", "none"] = Field(
        "minhash",
        description="minhash links paraphrased near-duplicates (~9 GB / 2.68M passages); "
                    "exact links verbatim copies only and is what full-corpus runs use")
    burst_window_hours: float = Field(24.0, gt=0)

    @field_validator("chunk_overlap_words")
    @classmethod
    def _overlap_lt_chunk(cls, v: int, info):  # type: ignore[no-untyped-def]
        chunk = info.data.get("chunk_words", 100)
        if v >= chunk:
            raise ValueError("chunk_overlap_words must be smaller than chunk_words")
        return v


class EmbeddingConfig(BaseModel):
    backend: Literal["hashing", "huggingface"] = "hashing"
    model_name: str = "facebook/contriever"
    revision: Optional[str] = Field(
        None, description="Hugging Face commit SHA; None means the repository default branch")
    precision: str = Field("float32", description="torch dtype used to load the embedding model")
    dim: int = Field(256, ge=16, description="only used by the hashing backend")
    batch_size: int = Field(32, ge=1)
    device: str = "cpu"
    query_prefix: str = ""          # BGE wants "Represent this sentence...: " style prefixes
    document_prefix: str = ""
    normalize: bool = True


class IndexConfig(BaseModel):
    backend: Literal["numpy", "faiss"] = "numpy"
    faiss_kind: Literal["flat", "ivfpq", "hnsw"] = "flat"
    nlist: int = Field(4096, ge=1)
    pq_m: int = Field(96, ge=1)
    nbits: int = Field(8, ge=1, le=16)
    hnsw_m: int = Field(32, ge=4)
    nprobe: int = Field(16, ge=1)


class RetrievalConfig(BaseModel):
    candidate_pool: int = Field(50, ge=1, description="K' before trust re-ranking")
    top_k: int = Field(5, ge=1, description="k passages handed to the generator")
    trust_lambda: float = Field(1.0, ge=0.0, description="score = sim * t_eff ** lambda")
    trust_floor: float = Field(0.05, ge=0.0, le=1.0, description="cold-start fairness floor")
    exclude_blocked: bool = True


class SignalsConfig(BaseModel):
    neighbourhood_k: int = Field(10, ge=2)
    cluster_similarity_threshold: float = Field(0.8, ge=0.0, le=1.0)
    burst_window_hours: float = Field(24.0, gt=0)
    source_mature_days: float = Field(180.0, gt=0)
    source_mature_docs: int = Field(50, ge=1)
    duplicate_threshold: float = Field(0.95, ge=0.0, le=1.0, description="S6: near-identical neighbour")
    min_corpus_for_isolation: int = Field(10_000, ge=0,
        description="S6 uses the isolation tail only above this corpus size; below it, unique clean "
                    "passages would look isolated and generate false positives")
    enabled: Dict[str, bool] = Field(default_factory=dict)


class ScorerConfig(BaseModel):
    model_path: Optional[str] = None
    target_high_fpr: float = Field(0.02, gt=0.0, lt=1.0, description="clean passages in HIGH band")
    escalation_budget: float = Field(1.0, gt=0.0, description="mean MEDIUM passages per query")
    theta_low: float = Field(0.30, ge=0.0, le=1.0)
    theta_high: float = Field(0.75, ge=0.0, le=1.0)
    C: float = Field(1.0, gt=0.0)
    calibration: Literal["isotonic", "sigmoid", "none"] = "isotonic"
    random_state: int = 20260921

    @field_validator("theta_high")
    @classmethod
    def _ordered(cls, v: float, info):  # type: ignore[no-untyped-def]
        low = info.data.get("theta_low", 0.0)
        if v < low:
            raise ValueError("theta_high must be >= theta_low")
        return v


class GenerationConfig(BaseModel):
    backend: Literal["stub", "openai", "ollama", "huggingface"] = "stub"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    revision: Optional[str] = Field(
        None, description="model commit/revision requested from the serving stack; recorded in the manifest")
    precision: Optional[str] = Field(
        None, description="declared serving precision (for example bfloat16 or int4); recorded, not inferred")
    quantization: Optional[str] = Field(
        None, description="declared generator quantization (for example q4_K_M); recorded in the manifest")
    serving_version: Optional[str] = Field(
        None, description="expected runtime version of Ollama/vLLM; verified when exposed")
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    seed: int = 20260921
    max_tokens: int = Field(256, ge=16)
    max_context_docs: int = Field(5, ge=1)
    abstain_evidence_mass: float = Field(0.5, ge=0.0, description="tau_ans: min trust-weighted evidence mass; one neutral-trust source == 0.5, two independent neutral sources == 0.75")
    require_citations: bool = True
    check_citations: bool = False       # needs Person B's NLI verifier
    cache_path: Optional[str] = Field(None, description="SQLite file caching LLM outputs (temperature 0 only)")
    timeout_s: float = Field(120.0, gt=0)


class StorageConfig(BaseModel):
    root: str = "runs/default"
    provenance_db: str = "provenance.sqlite3"
    answer_db: str = "answers.sqlite3"
    index_path: str = "index"
    scorer_path: str = "scorer.joblib"
    trust_db: str = "trust.sqlite3"


class TimeConfig(BaseModel):
    """Which clock the system uses (see ``utils.clock``).

    ``wall`` is the production setting.  ``logical`` maps stream step *n* to
    ``epoch + n * step_seconds`` so simulation is deterministic and never reads
    the machine clock.
    """

    mode: Literal["wall", "logical"] = "wall"
    epoch: float = Field(1_700_000_000.0, description="POSIX seconds of simulation step 0")
    step_seconds: float = Field(3600.0, gt=0, description="simulated seconds per stream step")


class TrustSettings(BaseModel):
    """Configuration surface for the trust ledger (mirrors ``TrustConfig``).

    Kept separate from ``trust.ledger.TrustConfig`` so the YAML stays a plain
    data contract; ``security.build_trust_config`` maps one to the other.
    """

    hierarchical: bool = True
    support_doc_alpha: float = Field(1.0, ge=0.0)
    support_family_alpha: float = Field(0.5, ge=0.0)
    support_source_alpha: float = Field(0.3, ge=0.0)
    refute_doc_beta: float = Field(2.0, ge=0.0)
    refute_family_beta: float = Field(1.0, ge=0.0)
    refute_source_beta: float = Field(0.5, ge=0.0)
    prior_weight: float = Field(5.0, gt=0)
    kappa: float = Field(3.0, ge=0.0)
    t_cap: float = Field(0.6, ge=0.0, le=1.0)
    decay_gamma: float = Field(0.95, ge=0.0, le=1.0)
    decay_interval: int = Field(100, ge=0)
    quarantine_refutations: int = Field(2, ge=1)
    reject_refutations: int = Field(3, ge=1)
    recovery_threshold: float = Field(0.5, ge=0.0, le=1.0)
    quarantine_t_eff: float = Field(0.20, ge=0.0, le=1.0)
    monitored_t_eff: float = Field(0.60, ge=0.0, le=1.0)
    high_band_beta_penalty: float = Field(0.3, ge=0.0)
    burst_trust_discount: float = Field(0.3, ge=0.0, le=1.0)
    burst_window_seconds: float = Field(60.0, gt=0)
    burst_min_docs: int = Field(4, ge=1)
    cold_start_influence: float = Field(0.25, ge=0.0, le=1.0)
    source_age_ramp_hours: float = Field(24.0, ge=0.0)
    cold_start_min_observations: int = Field(5, ge=1)
    record_history: bool = True

    @field_validator("reject_refutations")
    @classmethod
    def _reject_after_quarantine(cls, v: int, info):  # type: ignore[no-untyped-def]
        quarantine = info.data.get("quarantine_refutations", 1)
        if v < quarantine:
            raise ValueError("reject_refutations must be >= quarantine_refutations")
        return v

    @field_validator("quarantine_t_eff")
    @classmethod
    def _threshold_order(cls, v: float, info):  # type: ignore[no-untyped-def]
        monitored = info.data.get("monitored_t_eff")
        if monitored is not None and v > monitored:
            raise ValueError("quarantine_t_eff must be <= monitored_t_eff")
        return v


class SecurityConfig(BaseModel):
    """Person B's defence stack.  Enabled by default: the protected
    configuration must be the safe default, not an opt-in demo."""

    enabled: bool = True
    trust_db: Optional[str] = None        # None -> storage.root / storage.trust_db
    queue_size: int = Field(1000, ge=1)
    max_corroboration: int = Field(3, ge=1)
    verifier: Literal["auto", "nli", "lexical"] = "lexical"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_revision: Optional[str] = Field(
        None, description="Hugging Face commit SHA for the NLI model; None means the default branch")
    drain_queue: bool = True              # verify queued HIGH-band items after each answer
    fail_open_without_verifier: bool = False
    # MEDIUM + NEUTRAL means "corroboration did not resolve this".  The default
    # is to withhold it from the answer (reversible) rather than treat silence
    # as support.  Set to "use" only for an explicit research ablation.
    medium_neutral_action: Literal["exclude", "use"] = "exclude"
    min_refute_topical_overlap: int = Field(1, ge=1)
    min_refute_topic_coverage: float = Field(0.50, ge=0.0, le=1.0)
    min_refute_query_coverage: float = Field(0.60, ge=0.0, le=1.0)
    support_threshold: float = Field(0.50, ge=0.0, le=1.0)
    refute_threshold: float = Field(0.10, ge=0.0, le=1.0)
    max_source_mass: float = Field(0.6, gt=0.0, le=1.0)
    counterfactual_influence: bool = True
    max_verifier_llm_calls: int = Field(2, ge=1, description="NOT enforced: the verifier makes at most two LLM calls by construction (single-document answer + counterfactual); kept for config compatibility")
    probe_timeout_s: float = Field(
        10.0, gt=0.0, le=120.0,
        description="timeout for the functional model-readiness probes, in seconds",
    )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionaries into a copy of ``base``.

    Config sections are nested models.  Replacing a section with a partial
    override (for example ``security={"enabled": False}``) must not silently
    reset every other field in that section to its default.  The experiment
    runner relies on this: it changes only ``security.enabled`` for baselines,
    but must retain the configured verifier, model revision and trust settings.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


class Config(BaseModel):
    seed: int = 20260921
    ingestion: IngestionConfig = IngestionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    signals: SignalsConfig = SignalsConfig()
    scorer: ScorerConfig = ScorerConfig()
    generation: GenerationConfig = GenerationConfig()
    storage: StorageConfig = StorageConfig()
    time: TimeConfig = TimeConfig()
    trust: TrustSettings = TrustSettings()
    security: SecurityConfig = SecurityConfig()

    @classmethod
    def load(cls, path: Optional[str | Path] = None, **overrides: Any) -> "Config":
        data: Dict[str, Any] = {}
        if path is not None:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        data = _deep_merge(data, overrides)
        return cls.model_validate(data)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)

    def resolve_path(self, *parts: str) -> Path:
        """Return the effective path without creating anything on disk.

        ``path`` is the runtime helper and creates the parent directory.  Read-only
        diagnostics must use this method so a preflight cannot leave an empty
        storage tree behind in an experiment directory.
        """
        return Path(self.storage.root).joinpath(*parts)

    def path(self, *parts: str) -> Path:
        p = self.resolve_path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def trust_db_path(self, create_parent: bool = True) -> str:
        """Path of the persistent trust ledger (``:memory:`` stays in-memory).

        ``create_parent=False`` is for read-only checks such as the preflight.
        """
        configured = self.security.trust_db or self.storage.trust_db
        if configured == ":memory:":
            return ":memory:"
        if create_parent:
            return str(self.path(configured))
        return str(self.resolve_path(configured))
