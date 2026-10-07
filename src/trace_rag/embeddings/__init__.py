from typing import TYPE_CHECKING

from ..config import EmbeddingConfig
from .base import BaseEmbedder, l2_normalise
from .hashing_embedder import HashingEmbedder

if TYPE_CHECKING:  # pragma: no cover
    from .hf_embedder import HFEmbedder


def build_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Factory driven by config, so swapping Contriever for BGE is a YAML edit."""
    if config.backend == "hashing":
        return HashingEmbedder(dim=config.dim, normalize=config.normalize)
    if config.backend == "huggingface":
        from .hf_embedder import HFEmbedder

        pooling = "cls" if "bge" in config.model_name.lower() else "mean"
        return HFEmbedder(model_name=config.model_name, device=config.device, pooling=pooling,
                          max_length=config.max_length, dtype=config.dtype,
                          query_prefix=config.query_prefix, document_prefix=config.document_prefix,
                          normalize=config.normalize)
    raise ValueError(f"unknown embedding backend: {config.backend}")


__all__ = ["BaseEmbedder", "HashingEmbedder", "build_embedder", "l2_normalise"]
