"""Word-window chunking that preserves provenance and is byte-for-byte deterministic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Sequence

from ..utils.textnorm import normalise

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    n_words: int
    start_word: int
    end_word: int


def chunk_text(text: str, chunk_words: int = 100, overlap_words: int = 20,
               min_words: int = 15) -> List[Chunk]:
    """Split ``text`` into overlapping word windows.

    ~100 words matches the passage granularity of the BEIR corpora, so chunks
    from real PDFs are comparable with benchmark passages.  A trailing window
    shorter than ``min_words`` is merged into the previous chunk instead of
    being emitted as a fragment.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    words = _WORD.findall(normalise(text))
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks: List[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        window = words[start:end]
        if len(window) < min_words and chunks:
            prev = chunks[-1]
            merged_words = words[prev.start_word:end]
            chunks[-1] = Chunk(prev.ordinal, " ".join(merged_words), len(merged_words),
                               prev.start_word, end)
            break
        chunks.append(Chunk(len(chunks), " ".join(window), len(window), start, end))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_passage(text: str, chunk_words: int = 100, overlap_words: int = 20,
                  min_words: int = 15) -> List[Chunk]:
    """BEIR passages are already passage-sized: keep them whole unless oversized."""
    words = _WORD.findall(normalise(text))
    if len(words) <= chunk_words * 1.5:
        return [Chunk(0, " ".join(words), len(words), 0, len(words))] if words else []
    return chunk_text(text, chunk_words, overlap_words, min_words)
