"""Ingestion pipeline: files or BEIR JSONL -> provenance store -> chunks ready to embed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from ..config import IngestionConfig
from ..utils.hashing import FamilyAssigner, sha256_text
from ..utils.logging import get_logger
from ..utils.textnorm import normalise
from .chunking import chunk_passage, chunk_text
from .parsers import iter_beir_corpus, iter_files, parse_file
from .provenance import ChunkRecord, ProvenanceStore, make_chunk_id

logger = get_logger(__name__)


@dataclass
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    families: int = 0
    errors: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, object]:
        return {"documents": self.documents, "chunks": self.chunks, "skipped": self.skipped,
                "families": self.families, "errors": list(self.errors)}


class Ingestor:
    """Turns raw documents into provenance-carrying chunks.

    Every chunk leaves this class with: doc_id, chunk_id, source_id, family_id,
    SHA-256, word count and ingestion timestamp.  Nothing downstream is allowed
    to invent those fields.
    """

    def __init__(self, store: ProvenanceStore, config: Optional[IngestionConfig] = None) -> None:
        self.store = store
        self.config = config or IngestionConfig()
        self.families = FamilyAssigner(
            num_perm=self.config.minhash_perm,
            bands=self.config.minhash_bands,
            shingle_width=self.config.shingle_width,
            threshold=self.config.family_threshold,
        )

    # ----------------------------------------------------------------- public
    def ingest_text(self, doc_id: str, text: str, source_id: str, title: str = "",
                    origin: str = "", ingested_at: Optional[float] = None,
                    passage_mode: bool = False, metadata: Optional[Dict[str, object]] = None
                    ) -> List[ChunkRecord]:
        text = normalise(text)
        if not text:
            return []
        ts = time.time() if ingested_at is None else float(ingested_at)
        splitter = chunk_passage if passage_mode else chunk_text
        pieces = splitter(text, self.config.chunk_words, self.config.chunk_overlap_words,
                          self.config.min_chunk_words)
        records: List[ChunkRecord] = []
        for piece in pieces:
            chunk_id = make_chunk_id(doc_id, piece.ordinal)
            family_id = self.families.assign(chunk_id, piece.text)
            records.append(ChunkRecord(
                chunk_id=chunk_id, doc_id=doc_id, source_id=source_id, family_id=family_id,
                ordinal=piece.ordinal, text=piece.text, n_words=piece.n_words,
                sha256=sha256_text(piece.text), ingested_at=ts,
            ))
        if records:
            self.store.add_document(doc_id=doc_id, source_id=source_id, text_sha256=sha256_text(text),
                                    chunks=records, title=title, origin=origin, ingested_at=ts,
                                    metadata=metadata or {})
        return records

    def ingest_paths(self, root: str | Path, source_id_fn: Optional[Callable[[Path], str]] = None,
                     recursive: bool = True) -> IngestionReport:
        """Ingest every supported file under ``root``.

        ``source_id_fn`` maps a path to a contributor id; the default uses the
        immediate parent directory, which is the usual shared-drive layout.
        """
        report = IngestionReport()
        source_id_fn = source_id_fn or (lambda p: p.parent.name or "root")
        for path in iter_files(root, recursive=recursive):
            try:
                text = parse_file(path)
            except Exception as exc:                      # keep going, record the failure
                report.skipped += 1
                report.errors.append(f"{path}: {exc}")
                logger.warning("skipping %s: %s", path, exc)
                continue
            if not text:
                report.skipped += 1
                continue
            doc_id = f"file::{path.name}::{sha256_text(str(path.resolve()))[:12]}"
            records = self.ingest_text(
                doc_id=doc_id, text=text, source_id=source_id_fn(path), title=path.stem,
                origin=str(path), ingested_at=path.stat().st_mtime,
            )
            report.documents += 1
            report.chunks += len(records)
        report.families = len(self.families.family_sizes)
        return report

    def ingest_beir(self, corpus_path: str | Path, source_assigner: "SourceAssigner",
                    limit: Optional[int] = None, start_time: Optional[float] = None,
                    seconds_per_doc: float = 0.0) -> IngestionReport:
        """Ingest a BEIR ``corpus.jsonl``.

        BEIR passages carry no contributor metadata, so ``source_assigner``
        supplies simulated provenance (plan Section 6.1).  ``seconds_per_doc``
        spreads ingestion timestamps so that the burst signal has a realistic
        baseline instead of every clean passage sharing one timestamp.
        """
        report = IngestionReport()
        base = time.time() if start_time is None else float(start_time)
        for i, row in enumerate(iter_beir_corpus(corpus_path)):
            if limit is not None and i >= limit:
                break
            text = " ".join(part for part in (row["title"], row["text"]) if part)
            records = self.ingest_text(
                doc_id=row["doc_id"], text=text, source_id=source_assigner.assign(row["doc_id"]),
                title=row["title"], origin=str(corpus_path), passage_mode=True,
                ingested_at=base + i * seconds_per_doc,
            )
            if not records:
                report.skipped += 1
                continue
            report.documents += 1
            report.chunks += len(records)
        report.families = len(self.families.family_sizes)
        return report


class SourceAssigner:
    """Deterministic simulated provenance for corpora without contributor metadata.

    Contribution counts follow a Zipf-like distribution, which is what real
    shared drives look like: a few heavy contributors and a long tail.  The
    mapping is a pure function of ``doc_id`` and ``seed``, so two machines
    running the same config produce identical assignments.
    """

    def __init__(self, n_sources: int = 2000, seed: int = 20260921, zipf_exponent: float = 1.1,
                 prefix: str = "src") -> None:
        if n_sources <= 0:
            raise ValueError("n_sources must be positive")
        import numpy as np

        self.n_sources = int(n_sources)
        self.seed = int(seed)
        self.prefix = prefix
        ranks = np.arange(1, self.n_sources + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, float(zipf_exponent))
        self._cdf = np.cumsum(weights / weights.sum())

    def assign(self, doc_id: str) -> str:
        import numpy as np

        digest = sha256_text(f"{self.seed}:{doc_id}")
        u = int(digest[:16], 16) / float(1 << 64)
        idx = int(np.searchsorted(self._cdf, u, side="left"))
        idx = min(idx, self.n_sources - 1)
        return f"{self.prefix}_{idx:06d}"


class FixedSourceAssigner:
    """Assigns every document to one source (used for attacker corpora)."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    def assign(self, doc_id: str) -> str:
        return self.source_id
