"""Retroactive remediation: when a passage is quarantined, flag the answers it produced.

Person B calls :meth:`RemediationService.on_quarantine` from the quarantine
state machine.  Person C reads the resulting metrics (exposure window,
remediation recall) straight out of the answer log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .answer_log import AnswerLog, StoredAnswer


@dataclass(frozen=True)
class RemediationReport:
    doc_ids: tuple
    reason: str
    affected_answer_ids: tuple
    newly_flagged: int
    already_flagged: int
    exposure_window: int          # non-abstained answers served before quarantine
    affected_queries: tuple

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_ids": list(self.doc_ids), "reason": self.reason,
            "affected_answer_ids": list(self.affected_answer_ids),
            "newly_flagged": self.newly_flagged, "already_flagged": self.already_flagged,
            "exposure_window": self.exposure_window, "affected_queries": list(self.affected_queries),
        }


class RemediationService:
    def __init__(self, answer_log: AnswerLog, roles: Sequence[str] = ("cited", "context")) -> None:
        self.answer_log = answer_log
        self.roles = tuple(roles)

    def on_quarantine(self, doc_ids: Sequence[str], reason: str = "document quarantined",
                      at: Optional[float] = None, dry_run: bool = False) -> RemediationReport:
        """Flag every past answer that used any of ``doc_ids``.

        ``dry_run=True`` reports without writing, which is what the dashboard
        uses to preview the blast radius before an administrator confirms.
        """
        doc_ids = tuple(dict.fromkeys(doc_ids))
        affected: List[StoredAnswer] = self.answer_log.answers_using(doc_ids, roles=self.roles)
        already = sum(1 for a in affected if a.flagged)
        to_flag = [a.answer_id for a in affected if not a.flagged]
        newly = 0 if dry_run else self.answer_log.flag(to_flag, reason, at)
        exposure = sum(1 for a in affected if not a.abstained)
        return RemediationReport(
            doc_ids=doc_ids, reason=reason,
            affected_answer_ids=tuple(a.answer_id for a in affected),
            newly_flagged=int(newly), already_flagged=int(already),
            exposure_window=int(exposure),
            affected_queries=tuple(sorted({a.query_id for a in affected})),
        )

    def remediation_recall(self, poisoned_answer_ids: Sequence[int]) -> Optional[float]:
        """Fraction of known-poisoned answers that remediation actually flagged.

        Person C supplies the ground-truth ids from the attack harness; this
        function does not know which answers were poisoned on its own.
        """
        ids = list(dict.fromkeys(int(i) for i in poisoned_answer_ids))
        if not ids:
            return None
        flagged = 0
        for answer_id in ids:
            stored = self.answer_log.get(answer_id)
            if stored is not None and stored.flagged:
                flagged += 1
        return float(flagged / len(ids))
