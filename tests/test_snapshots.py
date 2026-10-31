from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from trace_rag import snapshots

SIGNATURE = "sha256:" + "a" * 64
OTHER_SIGNATURE = "sha256:" + "b" * 64


def _write_run(run: Path, steps: int, value: int = 0, signature: str = SIGNATURE,
               index_bytes: bytes = b"index-v1", index_base: str | None = None,
               create_stream: bool = True) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    database = run / "trust.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS state (value INTEGER)")
        connection.execute("DELETE FROM state")
        connection.execute("INSERT INTO state VALUES (?)", (value,))
        connection.commit()
    finally:
        connection.close()
    base = run / (index_base or "index")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".npz").write_bytes(index_bytes)
    (run / "run_manifest.json").write_text(
        json.dumps({"schema": "ragledger.run_manifest.v2", "state": "running",
                    "signature": signature}), encoding="utf-8")
    if create_stream:
        rows = [json.dumps({"step": index}) for index in range(steps)]
        (run / "stream.jsonl").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    (run / "checkpoint.json").write_text(json.dumps({
        "schema": "ragledger.checkpoint.v1", "signature": signature,
        "completed_steps": steps, "in_flight_step": None,
    }), encoding="utf-8")
    return database


def _snapshot(run: Path, backup: Path, steps: int, copy_index: bool = False,
              value: int = 0, signature: str = SIGNATURE,
              index_base: str | None = None) -> dict:
    database = _write_run(run, steps, value=value, signature=signature,
                          index_base=index_base)
    base = str(run / (index_base or "index"))
    return snapshots.backup_snapshot(run, backup, steps, [database], base,
                                     copy_index=copy_index)


def _step_dirs(backup: Path) -> list[Path]:
    return sorted(item for item in backup.iterdir()
                  if item.is_dir() and item.name.startswith("step-"))


def _checkpoint(path: Path) -> dict:
    return json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))


def _value(database: Path) -> int:
    connection = sqlite3.connect(database)
    try:
        return int(connection.execute("SELECT value FROM state").fetchone()[0])
    finally:
        connection.close()


def test_snapshot_at_step_zero_and_restore_without_active_directory(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    meta = _snapshot(run, backup, 0, copy_index=True, value=7)
    assert meta["completed_steps"] == 0
    shutil.rmtree(run)

    restored = snapshots.restore_latest_snapshot(run, backup,
                                                  expected_signature=SIGNATURE)

    assert restored is not None and restored["completed_steps"] == 0
    assert _checkpoint(run)["completed_steps"] == 0
    assert _value(run / "trust.sqlite3") == 7
    assert (run / "index.npz").read_bytes() == b"index-v1"


def test_snapshot_after_multiple_steps_and_restore_latest(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    late = _snapshot(run, backup, 3, value=3)
    assert late["completed_steps"] == 3

    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 1, "in_flight_step": 1}))
    restored = snapshots.restore_latest_snapshot(run, backup)

    assert restored is not None and restored["completed_steps"] == 3
    assert _value(run / "trust.sqlite3") == 3
    assert len((run / "stream.jsonl").read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.parametrize("corruption", ["checkpoint", "stream", "sqlite", "manifest"])
def test_corrupt_snapshot_is_skipped_and_previous_valid_snapshot_wins(tmp_path, corruption):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    _snapshot(run, backup, 2, value=2)
    latest = _step_dirs(backup)[-1]

    if corruption == "checkpoint":
        (latest / "checkpoint.json").write_text("{not json", encoding="utf-8")
    elif corruption == "stream":
        with open(latest / "stream.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": 99}) + "\n")
    elif corruption == "sqlite":
        (latest / "trust.sqlite3").write_bytes(b"not a sqlite database")
    else:
        manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["signature"] = "different"
        (latest / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = snapshots.snapshot_summary(backup)
    assert summary["completed_steps"] == 1

    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 0, "in_flight_step": 0}))
    restored = snapshots.restore_latest_snapshot(run, backup)
    assert restored is not None and restored["completed_steps"] == 1
    assert _value(run / "trust.sqlite3") == 1


def test_corrupt_index_digest_blocks_restore(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    (backup / "index" / "index.npz").write_bytes(b"tampered")

    assert snapshots.snapshot_summary(backup) == {}
    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 0, "in_flight_step": 0}))
    assert snapshots.restore_latest_snapshot(run, backup) is None


def test_interrupted_snapshot_staging_is_ignored_and_cleaned(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    interrupted = backup / ".step-000002-interrupted.tmp"
    interrupted.mkdir()
    (interrupted / "snapshot.json").write_text("{}", encoding="utf-8")
    index_staging = backup / ".index-interrupted.tmp"
    index_staging.mkdir()

    summary = snapshots.snapshot_summary(backup)

    assert summary["completed_steps"] == 1
    assert not interrupted.exists()
    assert not index_staging.exists()


def test_interrupted_index_replacement_is_rolled_forward(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    os.replace(backup / "index", backup / ".index.previous")
    assert not (backup / "index").exists()

    summary = snapshots.snapshot_summary(backup)

    assert summary["completed_steps"] == 1
    assert (backup / "index" / "index.npz").exists()
    assert not (backup / ".index.previous").exists()


def test_replace_rolls_back_when_installing_new_directory_fails(tmp_path, monkeypatch):
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated reset during install")
        return real_replace(source, target)

    monkeypatch.setattr(snapshots.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated reset"):
        snapshots._replace(staging, destination)

    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / ".destination.previous").exists()


def test_restore_preserves_sqlite_and_index_state_after_runtime_reset(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 2, copy_index=True, value=11)
    original_index = (run / "index.npz").read_bytes()

    connection = sqlite3.connect(run / "trust.sqlite3")
    connection.execute("UPDATE state SET value = 99")
    connection.commit()
    connection.close()
    (run / "index.npz").write_bytes(b"active-index-mutated")
    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 2, "in_flight_step": 2}))

    restored = snapshots.restore_latest_snapshot(run, backup)

    assert restored is not None and restored["completed_steps"] == 2
    assert _value(run / "trust.sqlite3") == 11
    assert (run / "index.npz").read_bytes() == original_index
    assert _checkpoint(run)["in_flight_step"] is None


def test_active_checkpoint_newer_than_snapshot_is_left_alone(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 2, copy_index=True, value=2)
    _write_run(run, 4, value=4)
    before = (run / "checkpoint.json").read_text(encoding="utf-8")

    assert snapshots.restore_latest_snapshot(run, backup) is None
    assert (run / "checkpoint.json").read_text(encoding="utf-8") == before
    assert _value(run / "trust.sqlite3") == 4


def test_recovery_is_idempotent_and_does_not_mix_signatures(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 2, copy_index=True, value=2)
    shutil.rmtree(run)

    first = snapshots.restore_latest_snapshot(run, backup,
                                               expected_signature=SIGNATURE)
    second = snapshots.restore_latest_snapshot(run, backup,
                                                expected_signature=SIGNATURE)

    assert first is not None and second is None
    assert _checkpoint(run)["signature"] == SIGNATURE

    _write_run(run, 2, value=9, signature=OTHER_SIGNATURE)
    (run / "checkpoint.json").write_text(json.dumps({
        "signature": OTHER_SIGNATURE, "completed_steps": 2,
        "in_flight_step": 2}))
    assert snapshots.restore_latest_snapshot(
        run, backup, expected_signature=SIGNATURE) is None
    assert _checkpoint(run)["signature"] == OTHER_SIGNATURE


def test_wrong_expected_signature_is_rejected_before_active_files_change(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 2, copy_index=True, value=5)

    # Make the active run interrupted and inconsistent with the snapshot.  A
    # restore with the wrong expected signature must not touch any of it.
    connection = sqlite3.connect(run / "trust.sqlite3")
    connection.execute("UPDATE state SET value = 9")
    connection.commit()
    connection.close()
    (run / "index.npz").write_bytes(b"active-index")
    checkpoint_before = json.dumps({
        "signature": SIGNATURE, "completed_steps": 2, "in_flight_step": 2,
    })
    (run / "checkpoint.json").write_text(checkpoint_before, encoding="utf-8")

    assert snapshots.restore_latest_snapshot(
        run, backup, expected_signature=OTHER_SIGNATURE) is None
    assert (run / "checkpoint.json").read_text(encoding="utf-8") == checkpoint_before
    assert _value(run / "trust.sqlite3") == 9
    assert (run / "index.npz").read_bytes() == b"active-index"

    # The correct snapshot is still recoverable after that failed attempt.
    restored = snapshots.restore_latest_snapshot(run, backup,
                                                  expected_signature=SIGNATURE)
    assert restored is not None and restored["completed_steps"] == 2
    assert _value(run / "trust.sqlite3") == 5
    assert (run / "index.npz").read_bytes() == b"index-v1"
    assert _checkpoint(run)["in_flight_step"] is None


def test_restore_without_an_experiment_identity_is_refused(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    shutil.rmtree(run)

    assert snapshots.restore_latest_snapshot(run, backup) is None
    assert not (run / "checkpoint.json").exists()


def test_completed_run_is_never_replaced_by_a_snapshot(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1)
    (run / "metrics.json").write_text('{"state": "completed"}', encoding="utf-8")
    checkpoint_before = (run / "checkpoint.json").read_text(encoding="utf-8")

    assert snapshots.restore_latest_snapshot(
        run, backup, expected_signature=SIGNATURE) is None
    assert (run / "metrics.json").read_text(encoding="utf-8") == '{"state": "completed"}'
    assert (run / "checkpoint.json").read_text(encoding="utf-8") == checkpoint_before


def test_shared_backup_parent_cannot_mix_experiment_state(tmp_path):
    parent = tmp_path / "drive" / "backups"
    run_a, run_b = tmp_path / "active" / "pilot", tmp_path / "active" / "research"
    _snapshot(run_a, parent, 1, copy_index=True, value=1, signature=SIGNATURE)

    # A second experiment cannot replace the shared prepared index or add a
    # snapshot under an index that belongs to the first experiment.
    with pytest.raises(ValueError, match="different run signature"):
        _snapshot(run_b, parent, 1, copy_index=True, value=2,
                  signature=OTHER_SIGNATURE)
    summary = snapshots.snapshot_summary(parent)
    assert summary["run_signature"] == SIGNATURE
    assert summary["completed_steps"] == 1

    # Nor can the other experiment's expected signature overwrite A's active run.
    checkpoint_before = (run_a / "checkpoint.json").read_text(encoding="utf-8")
    assert snapshots.restore_latest_snapshot(
        run_a, parent, expected_signature=OTHER_SIGNATURE) is None
    assert (run_a / "checkpoint.json").read_text(encoding="utf-8") == checkpoint_before
    assert _value(run_a / "trust.sqlite3") == 1

    # A remains recoverable, and B works in its own backup directory.
    shutil.rmtree(run_a)
    assert snapshots.restore_latest_snapshot(
        run_a, parent, expected_signature=SIGNATURE)["completed_steps"] == 1
    backup_b = tmp_path / "drive" / "backups_research"
    _snapshot(run_b, backup_b, 1, copy_index=True, value=2,
              signature=OTHER_SIGNATURE)
    shutil.rmtree(run_b)
    assert snapshots.restore_latest_snapshot(
        run_b, backup_b, expected_signature=OTHER_SIGNATURE)["completed_steps"] == 1


def test_restore_failure_before_checkpoint_commit_can_be_retried(tmp_path, monkeypatch):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 2, copy_index=True, value=2)
    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 2, "in_flight_step": 2}))

    real_install = snapshots._install_file

    def fail_at_checkpoint(source, destination):
        if Path(destination).name == "checkpoint.json":
            raise OSError("simulated runtime reset before checkpoint commit")
        return real_install(source, destination)

    monkeypatch.setattr(snapshots, "_install_file", fail_at_checkpoint)
    with pytest.raises(OSError, match="simulated runtime reset"):
        snapshots.restore_latest_snapshot(run, backup,
                                          expected_signature=SIGNATURE)
    assert _checkpoint(run)["in_flight_step"] == 2

    monkeypatch.setattr(snapshots, "_install_file", real_install)
    restored = snapshots.restore_latest_snapshot(run, backup,
                                                 expected_signature=SIGNATURE)
    assert restored is not None and restored["completed_steps"] == 2
    assert _checkpoint(run)["in_flight_step"] is None


def test_nested_index_path_is_restored_in_place(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "backup"
    _snapshot(run, backup, 1, copy_index=True, value=1,
              index_base="artifacts/index")
    assert (run / "artifacts" / "index.npz").exists()
    shutil.rmtree(run / "artifacts")
    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 1, "in_flight_step": 1}))

    restored = snapshots.restore_latest_snapshot(run, backup)

    assert restored is not None
    assert (run / "artifacts" / "index.npz").read_bytes() == b"index-v1"


def test_restore_uses_the_runner_database_and_index_layout(tmp_path):
    """Exercise the three SQLite files and index base the runner actually uses."""
    run, backup = tmp_path / "active" / "pilot", tmp_path / "drive" / "pilot"
    _write_run(run, 2, value=2)

    # run_trust_stream.py snapshots provenance, answers and trust databases.
    for name in ("provenance.sqlite3", "answers.sqlite3"):
        connection = sqlite3.connect(run / name)
        try:
            connection.execute("CREATE TABLE rows (value INTEGER)")
            connection.execute("INSERT INTO rows VALUES (2)")
            connection.commit()
        finally:
            connection.close()
    (run / "index.meta.json").write_text('{"dim": 4}', encoding="utf-8")
    databases = [run / "provenance.sqlite3", run / "answers.sqlite3",
                 run / "trust.sqlite3"]
    meta = snapshots.backup_snapshot(run, backup, 2, databases, str(run / "index"),
                                     copy_index=True)
    assert meta["sqlite_files"] == ["provenance.sqlite3", "answers.sqlite3",
                                    "trust.sqlite3"]
    assert meta["index_files"] == ["index.meta.json", "index.npz"]

    for name in ("provenance.sqlite3", "answers.sqlite3", "trust.sqlite3"):
        connection = sqlite3.connect(run / name)
        try:
            connection.execute("DELETE FROM rows" if name != "trust.sqlite3"
                               else "DELETE FROM state")
            if name != "trust.sqlite3":
                connection.execute("INSERT INTO rows VALUES (99)")
            else:
                connection.execute("INSERT INTO state VALUES (99)")
            connection.commit()
        finally:
            connection.close()
    (run / "index.npz").write_bytes(b"mutated")
    (run / "checkpoint.json").write_text(json.dumps({
        "signature": SIGNATURE, "completed_steps": 2, "in_flight_step": 2}))

    restored = snapshots.restore_latest_snapshot(run, backup)

    assert restored is not None
    for name in ("provenance.sqlite3", "answers.sqlite3"):
        connection = sqlite3.connect(run / name)
        try:
            assert connection.execute("SELECT value FROM rows").fetchone()[0] == 2
        finally:
            connection.close()
    assert _value(run / "trust.sqlite3") == 2
    assert (run / "index.npz").read_bytes() == b"index-v1"
    assert (run / "index.meta.json").read_text(encoding="utf-8") == '{"dim": 4}'
