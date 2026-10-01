from ..config import IndexConfig
from .numpy_index import IndexLoadError, NumpyFlatIndex


def build_index(config: IndexConfig, dim: int):  # type: ignore[no-untyped-def]
    if config.backend == "numpy":
        return NumpyFlatIndex(dim)
    if config.backend == "faiss":
        from .faiss_index import FaissIndex

        return FaissIndex(dim=dim, kind=config.faiss_kind, nlist=config.nlist, pq_m=config.pq_m,
                          nbits=config.nbits, hnsw_m=config.hnsw_m, nprobe=config.nprobe)
    raise ValueError(f"unknown index backend: {config.backend}")


def load_index(config: IndexConfig, path: str):  # type: ignore[no-untyped-def]
    if config.backend == "numpy":
        return NumpyFlatIndex.load(path)
    from .faiss_index import FaissIndex

    return FaissIndex.load(path)


__all__ = ["NumpyFlatIndex", "IndexLoadError", "build_index", "load_index"]
