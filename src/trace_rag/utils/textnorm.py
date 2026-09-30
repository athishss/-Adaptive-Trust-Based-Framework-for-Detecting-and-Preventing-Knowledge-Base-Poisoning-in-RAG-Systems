"""Deterministic text normalisation shared by ingestion, signals and citations."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence, Tuple

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
# Sentence split good enough for citation checking: end punctuation followed by
# whitespace + capital/digit, without breaking common abbreviations.
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "fig", "no"}
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def normalise(text: str) -> str:
    """NFKC-normalise, collapse whitespace, strip."""
    text = unicodedata.normalize("NFKC", text or "")
    return _WS.sub(" ", text).strip()


def tokenise(text: str) -> List[str]:
    """Lowercase alphanumeric tokens.  Deterministic, no external tokenizer."""
    return _TOKEN.findall((text or "").lower())


def shingles(tokens: Sequence[str], width: int = 5) -> List[Tuple[str, ...]]:
    """Overlapping token n-grams, used for near-duplicate detection."""
    if width <= 0:
        raise ValueError("width must be positive")
    if len(tokens) < width:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)]


def sentences(text: str) -> List[str]:
    """Split into sentences; keeps ordering and drops empties."""
    text = normalise(text)
    if not text:
        return []
    parts = _SENT_END.split(text)
    merged: List[str] = []
    for part in parts:
        if merged:
            prev_tokens = tokenise(merged[-1])
            last = prev_tokens[-1] if prev_tokens else ""
            if last in _ABBREV or (len(last) == 1 and merged[-1].rstrip().endswith(".")):
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    return [s.strip() for s in merged if s.strip()]


def word_count(text: str) -> int:
    return len(tokenise(text))
