"""Typed configuration.  Every threshold in the plan lives here, never in code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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
    dim: int = Field(256, ge=16, description="only used by the hashing backend")
    batch_size: int = Field(32, ge=1)
    max_length: int = Field(512, ge=8, le=4096)
    device: str = "cpu"              # cpu | auto | cuda | tpu (PyTorch/XLA)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
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
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    device: str = "auto"              # for the local Hugging Face generation backend
    temperature: float = 0.0
    seed: int = 20260921
    max_tokens: int = Field(256, ge=16)
    max_context_docs: int = Field(5, ge=1)
    abstain_evidence_mass: float = Field(0.5, ge=0.0, description="tau_ans: min trust-weighted evidence mass; one neutral-trust source == 0.5, two independent neutral sources == 0.75")
    require_citations: bool = True
    check_citations: bool = False       # fail-closed NLI check for each cited sentence
    citation_nli_threshold: float = Field(0.5, ge=0, le=1)
    timeout_s: float = Field(120.0, gt=0)


class TrustSettings(BaseModel):
    """Typed configuration for hierarchical trust, verification and policy."""

    enabled: bool = False
    w_s: float = Field(1.0, gt=0)
    w_r: float = Field(2.0, gt=0)
    w_max: float = Field(5.0, gt=0)
    support_family_alpha: float = Field(0.5, ge=0)
    support_source_alpha: float = Field(0.3, ge=0)
    refute_family_beta: float = Field(1.0, ge=0)
    refute_source_beta: float = Field(0.5, ge=0)
    prior_weight: float = Field(5.0, gt=0, description="m in the empirical-Bayes blend")
    kappa: float = Field(3.0, ge=0)
    t_cap: float = Field(0.6, ge=0, le=1)
    decay_gamma: float = Field(0.95, ge=0, le=1)
    decay_interval: int = Field(100, ge=1)
    quarantine_refutations: int = Field(2, ge=1)
    reject_refutations: int = Field(3, ge=1)
    hierarchical: bool = True
    recovery_threshold: float = Field(0.6, ge=0, le=1)
    quarantine_t_eff: float = Field(0.2, ge=0, le=1)
    monitored_t_eff: float = Field(0.6, ge=0, le=1)
    burst_trust_discount: float = Field(0.3, ge=0, le=1)
    burst_window_seconds: float = Field(60.0, gt=0)
    burst_min_docs: int = Field(4, ge=2)
    cold_start_influence: float = Field(0.25, ge=0, le=1)
    source_age_ramp_hours: float = Field(24.0, gt=0)
    cold_start_min_observations: int = Field(5, ge=1)
    record_history: bool = True
    queue_max_size: int = Field(1000, ge=1)
    verifier_max_corroboration: int = Field(3, ge=1)
    verifier_support_threshold: float = Field(0.15, ge=0, le=1)
    verifier_refute_threshold: float = Field(0.10, ge=0, le=1)
    verifier_min_mass: float = Field(0.20, ge=0, le=1)
    verifier_min_refute_topical_overlap: int = Field(1, ge=0)
    verifier_max_redundant_coverage: float = Field(0.80, ge=0, le=1)
    verifier_max_source_mass: float = Field(0.60, ge=0, le=1)
    verifier_counterfactual_influence: bool = True
    verifier_influence_agreement_threshold: float = Field(0.60, ge=0, le=1)
    nli_mode: Literal["auto", "nli", "lexical"] = "auto"

    @model_validator(mode="after")
    def _consistent_thresholds(self) -> "TrustSettings":
        if self.w_r <= self.w_s:
            raise ValueError("trust.w_r must be greater than trust.w_s")
        if not (self.quarantine_t_eff < self.monitored_t_eff):
            raise ValueError("trust.quarantine_t_eff must be below trust.monitored_t_eff")
        if self.reject_refutations <= self.quarantine_refutations:
            raise ValueError("trust.reject_refutations must exceed quarantine_refutations")
        if self.verifier_refute_threshold > self.verifier_support_threshold:
            raise ValueError("verifier refute threshold must not exceed support threshold")
        return self

    def to_ledger_config(self):  # type: ignore[no-untyped-def]
        """Build the Person B ledger config without importing B at module load."""
        from .trust.ledger import TrustConfig

        return TrustConfig(
            w_s=self.w_s, w_r=self.w_r, w_max=self.w_max,
            support_family_alpha=self.support_family_alpha,
            support_source_alpha=self.support_source_alpha,
            refute_family_beta=self.refute_family_beta,
            refute_source_beta=self.refute_source_beta,
            prior_weight=self.prior_weight, kappa=self.kappa, t_cap=self.t_cap,
            decay_gamma=self.decay_gamma, decay_interval=self.decay_interval,
            quarantine_refutations=self.quarantine_refutations,
            reject_refutations=self.reject_refutations,
            hierarchical=self.hierarchical, recovery_threshold=self.recovery_threshold,
            quarantine_t_eff=self.quarantine_t_eff,
            monitored_t_eff=self.monitored_t_eff,
            burst_trust_discount=self.burst_trust_discount,
            burst_window_seconds=self.burst_window_seconds,
            burst_min_docs=self.burst_min_docs,
            cold_start_influence=self.cold_start_influence,
            source_age_ramp_hours=self.source_age_ramp_hours,
            cold_start_min_observations=self.cold_start_min_observations,
            record_history=self.record_history,
        )

    def verifier_kwargs(self) -> Dict[str, Any]:
        return {
            "max_corroboration": self.verifier_max_corroboration,
            "support_threshold": self.verifier_support_threshold,
            "refute_threshold": self.verifier_refute_threshold,
            "min_mass": self.verifier_min_mass,
            "min_refute_topical_overlap": self.verifier_min_refute_topical_overlap,
            "max_redundant_coverage": self.verifier_max_redundant_coverage,
            "max_source_mass": self.verifier_max_source_mass,
            "counterfactual_influence": self.verifier_counterfactual_influence,
            "influence_agreement_threshold": self.verifier_influence_agreement_threshold,
            "use_nli": None if self.nli_mode == "auto" else self.nli_mode == "nli",
        }


class StorageConfig(BaseModel):
    root: str = "runs/default"
    provenance_db: str = "provenance.sqlite3"
    answer_db: str = "answers.sqlite3"
    trust_db: str = "trust.sqlite3"
    index_path: str = "index"
    scorer_path: str = "scorer.joblib"


class Config(BaseModel):
    seed: int = 20260921
    ingestion: IngestionConfig = IngestionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    signals: SignalsConfig = SignalsConfig()
    scorer: ScorerConfig = ScorerConfig()
    trust: TrustSettings = TrustSettings()
    generation: GenerationConfig = GenerationConfig()
    storage: StorageConfig = StorageConfig()

    @classmethod
    def load(cls, path: Optional[str | Path] = None, **overrides: Any) -> "Config":
        data: Dict[str, Any] = {}
        if path is not None:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        for key, value in overrides.items():
            data[key] = value
        return cls.model_validate(data)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)

    def path(self, *parts: str) -> Path:
        p = Path(self.storage.root).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
