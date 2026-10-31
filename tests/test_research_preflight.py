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

STREAM_SPEC = importlib.util.spec_from_file_location(
    "run_trust_stream_for_preflight", ROOT / "scripts" / "run_trust_stream.py")
assert STREAM_SPEC is not None and STREAM_SPEC.loader is not None
stream = importlib.util.module_from_spec(STREAM_SPEC)
STREAM_SPEC.loader.exec_module(stream)


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


def test_ledger_status_does_not_create_the_storage_root(tmp_path):
    root = tmp_path / "fresh" / "run"
    config = Config(storage={"root": str(root)})
    result = preflight.ledger_status(config)
    assert result["status"] == "ready"
    assert result["initialized"] is False
    assert not root.exists()


def test_storage_isolation_confines_absolute_and_escaping_paths(tmp_path):
    outside = tmp_path / "outside"
    config = Config(
        storage={
            "root": str(tmp_path / "run"),
            "provenance_db": str(outside / "provenance.sqlite3"),
            "answer_db": "state/answers.sqlite3",
            "index_path": "../escaped-index",
            "scorer_path": "scorer.joblib",
        },
        security={"trust_db": str(outside / "trust.sqlite3")},
    )
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    probe, rebased = preflight.isolate_storage_config(config, probe_root)

    assert config.storage.root == str(tmp_path / "run"), "isolation must not mutate the source"
    assert probe.storage.root == str(probe_root)
    assert set(rebased) == {"storage.provenance_db", "storage.index_path",
                            "security.trust_db"}
    effective = [
        probe.resolve_path(probe.storage.provenance_db),
        probe.resolve_path(probe.storage.answer_db),
        probe.resolve_path(probe.storage.index_path),
        probe.resolve_path(probe.storage.scorer_path),
        Path(probe.trust_db_path(create_parent=False)),
    ]
    for path in effective:
        assert preflight._is_within(path, probe_root), path
    assert not outside.exists()


def test_functional_doctor_cannot_contaminate_a_fresh_experiment_directory(tmp_path):
    """Regression: a real doctor writes SQLite files; preflight must isolate them."""
    run_dir = tmp_path / "active" / "pilot"
    config = Config(
        storage={"root": str(run_dir)},
        embedding={"backend": "hashing", "device": "cpu"},
        index={"backend": "numpy"},
        generation={"backend": "stub"},
        security={"enabled": True, "verifier": "lexical"},
    )

    result = preflight.functional_doctor(config, timeout=120.0,
                                         preflight_storage_root=tmp_path / "preflight")
    isolation = result["storage_isolation"]

    assert isolation["schema"] == preflight.PREFLIGHT_STORAGE_SCHEMA
    assert isolation["intended_root"] == str(run_dir)
    assert Path(isolation["probe_root"]).resolve() != run_dir.resolve()
    assert isolation["cleaned"] is True
    assert not Path(isolation["probe_root"]).exists()
    assert not run_dir.exists(), "preflight must not create the experiment directory"
    assert list((tmp_path / "preflight").iterdir()) == []
    # A fresh experiment directory that a preflight has not touched is a valid
    # starting point for the runner's output-state guard.
    assert stream.check_output_state(run_dir, resume=False) is None
