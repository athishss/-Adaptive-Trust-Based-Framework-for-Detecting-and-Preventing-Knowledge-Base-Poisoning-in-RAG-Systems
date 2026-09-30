"""Typed configuration.  Every threshold in the plan lives here, never in code."""

from __future__ import annotations

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
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    seed: int = 20260921
    max_tokens: int = Field(256, ge=16)
    max_context_docs: int = Field(5, ge=1)
    abstain_evidence_mass: float = Field(0.5, ge=0.0, description="tau_ans: min trust-weighted evidence mass; one neutral-trust source == 0.5, two independent neutral sources == 0.75")
    require_citations: bool = True
    check_citations: bool = False       # needs Person B's NLI verifier
    timeout_s: float = Field(120.0, gt=0)


class StorageConfig(BaseModel):
    root: str = "runs/default"
    provenance_db: str = "provenance.sqlite3"
    answer_db: str = "answers.sqlite3"
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
