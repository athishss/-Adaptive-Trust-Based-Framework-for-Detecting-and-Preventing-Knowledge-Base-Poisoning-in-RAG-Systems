"""Resource lifecycle regressions.

The audit found ``ResourceWarning: unclosed database`` from abandoned ledger,
store and answer-log objects.  These tests pin the cleanup contract: close is
idempotent, a constructor that fails after opening SQLite closes what it
opened, and pipeline shutdown can be repeated safely.
"""

from __future__ import annotations

import sqlite3

import pytest

from trace_rag.ingestion.provenance import ProvenanceStore
from trace_rag.pipeline import PersonAPipeline
from trace_rag.provenance.answer_log import AnswerLog
from trace_rag.trust.ledger import TrustLedger


class _FailingConnection:
    """Connection whose schema setup fails after ``connect`` succeeds."""

    row_factory = None

    def __init__(self) -> None:
        self.closed = False

    def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def executescript(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise sqlite3.OperationalError("schema setup failed")

    def close(self) -> None:
        self.closed = True


def test_sqlite_components_close_idempotently(tmp_path):
    resources = [
        TrustLedger(tmp_path / "trust.db"),
        ProvenanceStore(tmp_path / "prov.db"),
        AnswerLog(tmp_path / "answers.db"),
    ]
    for resource in resources:
        resource.close()
        resource.close()  # a repeated close must be a no-op, not an error


@pytest.mark.parametrize("factory", [
    lambda path: TrustLedger(path),
    lambda path: ProvenanceStore(path),
    lambda path: AnswerLog(path),
])
def test_failed_constructor_closes_its_connection(tmp_path, monkeypatch, factory):
    connection = _FailingConnection()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connection)

    with pytest.raises(sqlite3.OperationalError):
        factory(tmp_path / "broken.db")

    assert connection.closed is True


def test_pipeline_close_is_idempotent(config):
    pipeline = PersonAPipeline.from_config(config, load_existing_index=False)
    pipeline.close()
    pipeline.close()  # shutdown may be reached twice on an error path
