"""Chronological query-stream simulation for Person C evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StreamQuery:
    step: int
    query_id: str
    query: str
    target_query: Optional[str] = None
    is_target: bool = False

    def to_dict(self):
        return {
            "step": self.step,
            "query_id": self.query_id,
            "query": self.query,
            "target_query": self.target_query,
            "is_target": self.is_target,
        }


@dataclass(frozen=True)
class StreamIngestion:
    step: int
    doc_id: str
    text: str
    source_id: str
    family_id: Optional[str] = None
    is_poison: bool = False

    def to_dict(self):
        return {
            "step": self.step,
            "doc_id": self.doc_id,
            "text": self.text,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "is_poison": self.is_poison,
        }


@dataclass
class StreamResult:
    queries: List[StreamQuery]
    ingestions: List[StreamIngestion]
    answers: List[object]

    def to_dict(self):
        return {
            "queries": [q.to_dict() for q in self.queries],
            "ingestions": [i.to_dict() for i in self.ingestions],
            "answers": [
                a.to_dict() if hasattr(a, "to_dict") else a
                for a in self.answers
            ],
        }
