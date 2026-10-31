from __future__ import annotations

import importlib.util
from pathlib import Path

from trace_rag.config import Config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_research_environment", ROOT / "scripts" / "check_research_environment.py")
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def _check(status: str) -> dict:
    return {"name": "x", "status": status, "detail": "", "remediation": ""}


def test_static_failure_blocks_even_with_a_good_doctor():
    status, reason = preflight.classify_status(
        [_check("fail")], {"status": "PROTECTED"})
    assert status == preflight.BLOCKED
    assert "blocking" in reason


def test_static_success_without_functional_probe_is_only_degraded():
    status, _ = preflight.classify_status([_check("ok")], None)
    assert status == preflight.DEGRADED


def test_ready_requires_a_protected_functional_doctor():
    status, _ = preflight.classify_status(
        [_check("ok")], {"status": "PROTECTED"})
    assert status == preflight.READY


def test_unprotected_or_failed_doctor_cannot_be_ready():
    for report in ("UNPROTECTED", "failed", "UNKNOWN", None):
        status, _ = preflight.classify_status([_check("ok")], {"status": report})
        assert status == preflight.BLOCKED
    status, _ = preflight.classify_status([_check("ok")], {"status": "DEGRADED"})
    assert status == preflight.DEGRADED


def test_ledger_check_reports_an_uninitialized_writable_path(tmp_path):
    config = Config(storage={"root": str(tmp_path)})
    result = preflight.ledger_status(config)
    assert result["status"] == "ready"
    assert result["initialized"] is False
    assert not Path(config.trust_db_path()).exists()


def test_in_memory_ledger_is_rejected(tmp_path):
    config = Config(storage={"root": str(tmp_path)}, security={"trust_db": ":memory:"})
    result = preflight.ledger_status(config)
    assert result["status"] == "failed"
    assert "in-memory" in result["detail"]


def test_missing_dataset_files_are_reported(tmp_path):
    result = preflight.dataset_status(str(tmp_path / "missing.parquet"),
                                      str(tmp_path / "missing.jsonl"),
                                      str(tmp_path / "missing.tsv"))
    assert result["status"] == "failed"
    assert len(result["missing"]) == 3


def test_local_model_directory_is_detected_without_download(tmp_path):
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    result = preflight.model_cache_status(str(model))
    assert result["cached"] is True
    assert result["source"] == "local_path"


def test_stub_generation_does_not_require_an_http_service(tmp_path):
    config = Config(storage={"root": str(tmp_path)})
    result = preflight.endpoint_status(config)
    assert result["status"] == "not_applicable"
