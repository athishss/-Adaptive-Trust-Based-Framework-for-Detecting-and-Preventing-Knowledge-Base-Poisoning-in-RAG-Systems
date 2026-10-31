"""Atomic Drive snapshots for active Colab runs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCHEMA = "ragledger.drive_snapshot.v1"
SMALL_FILES = ("checkpoint.json", "run_manifest.json", "stream.jsonl",
               "metrics.json", "trust_history.csv")
KEEP = 8


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(source), timeout=30.0)
    dst = sqlite3.connect(str(destination))
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()


def _sqlite_ok(path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(path), timeout=5.0)
        try:
            return str(conn.execute("PRAGMA quick_check").fetchone()[0]).lower() == "ok"
        finally:
            conn.close()
    except Exception:
        return False


def _index_files(base: str) -> List[Path]:
    path = Path(base)
    return [item for item in (path.with_suffix(".faiss"), path.with_suffix(".meta.json"),
                              path.with_suffix(".npz")) if item.exists()]


def _index_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(directory.iterdir()):
        if item.is_file():
            digest.update(item.name.encode())
            digest.update(_sha(item).encode())
    return digest.hexdigest()


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _previous(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.previous"


def _recover_previous(destination: Path) -> None:
    """Roll back or finish an interrupted directory replacement.

    ``_replace`` never deletes the old directory before the new one is in place.
    If the process dies between the two renames, the old tree is still under
    ``.<name>.previous`` and is put back on the next snapshot/restore attempt.
    """
    previous = _previous(destination)
    if previous.exists() and not destination.exists():
        os.replace(previous, destination)
    elif previous.exists() and destination.exists():
        # The new tree was installed but the old copy was not removed.
        shutil.rmtree(previous, ignore_errors=True)


def _cleanup_temporary(backup_dir: Path) -> None:
    """Remove staging directories left by an interrupted snapshot attempt."""
    if not backup_dir.exists():
        return
    for item in backup_dir.iterdir():
        if item.is_dir() and item.name.startswith(".") and item.name.endswith(".tmp"):
            shutil.rmtree(item, ignore_errors=True)
    _recover_previous(backup_dir / "index")


def _replace(staging: Path, destination: Path) -> None:
    """Install *staging* at *destination* with a recoverable rollback point.

    A directory rename cannot be atomic across an existing directory, so the old
    tree is moved aside first.  If installing the new tree fails, the old tree is
    restored before the error propagates.  A process death between the renames
    leaves ``.<name>.previous`` for :func:`_recover_previous`.
    """
    _recover_previous(destination)
    if not destination.exists():
        os.replace(staging, destination)
        return
    previous = _previous(destination)
    if previous.exists():
        shutil.rmtree(previous, ignore_errors=True)
    os.replace(destination, previous)
    try:
        os.replace(staging, destination)
    except Exception:
        if not destination.exists() and previous.exists():
            os.replace(previous, destination)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def backup_snapshot(run_dir: Path, backup_dir: Path, completed_steps: int,
                    sqlite_paths: Sequence[Path], index_base: str,
                    copy_index: bool = False) -> Dict[str, Any]:
    """Snapshot a completed-step boundary atomically.

    SQLite files are copied with the SQLite backup API.  The checkpoint and
    stream must already describe exactly ``completed_steps``.
    """
    run_dir, backup_dir = Path(run_dir), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_temporary(backup_dir)
    checkpoint = _read(run_dir / "checkpoint.json")
    if int(checkpoint.get("completed_steps", -1)) != int(completed_steps):
        raise ValueError("checkpoint completed_steps differs from the snapshot step")
    if checkpoint.get("in_flight_step") is not None:
        raise ValueError("refusing to snapshot in-flight work")
    signature = str(checkpoint.get("signature") or "")
    if not signature:
        raise ValueError("checkpoint has no run signature; refusing an unidentifiable snapshot")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run_manifest.json is required before a snapshot can be trusted")
    manifest = _read(manifest_path)
    if str(manifest.get("signature") or "") != signature:
        raise ValueError("checkpoint and run_manifest signatures differ")
    stream = run_dir / "stream.jsonl"
    rows = len([line for line in stream.read_text(encoding="utf-8").splitlines()
                if line.strip()]) if stream.exists() else 0
    if rows != int(completed_steps):
        raise ValueError(f"stream has {rows} rows but checkpoint says {completed_steps}")
    index_dir = backup_dir / "index"
    index_manifest_path = backup_dir / "index_manifest.json"
    existing_index_manifest = _read(index_manifest_path) if index_manifest_path.exists() else {}
    existing_index_signature = str(existing_index_manifest.get("run_signature") or "")
    if existing_index_signature and existing_index_signature != signature:
        raise ValueError(
            "backup directory contains an index snapshot for a different run signature; "
            "use a separate --backup-dir for this experiment")
    if copy_index:
        staging_index = backup_dir / f".index-{_now()}.tmp"
        staging_index.mkdir(parents=True)
        files = _index_files(index_base)
        if not files:
            shutil.rmtree(staging_index, ignore_errors=True)
            raise FileNotFoundError(f"no index files found for {index_base}")
        for item in files:
            shutil.copy2(item, staging_index / item.name)
        _replace(staging_index, index_dir)
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError("index snapshot is missing; create the prepared snapshot first")
    index_files = sorted(item.name for item in index_dir.iterdir() if item.is_file())
    index_digest = _index_digest(index_dir)
    _write(index_manifest_path,
           {"digest": index_digest, "files": index_files, "run_signature": signature})
    # Store where the index lives inside the run so restore does not assume the
    # default top-level ``index`` name.
    run_resolved, base_resolved = run_dir.resolve(), Path(index_base).resolve()
    try:
        index_relative = base_resolved.relative_to(run_resolved).as_posix()
    except ValueError:
        index_relative = Path(index_base).name
    name = f"step-{int(completed_steps):06d}-{_now()}"
    staging = backup_dir / f".{name}.tmp"
    staging.mkdir(parents=True)
    copied: List[str] = []
    for filename in SMALL_FILES:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, staging / filename)
            copied.append(filename)
    sqlite_files: List[str] = []
    for source in sqlite_paths:
        source = Path(source)
        if not source.exists():
            continue
        try:
            relative = source.resolve().relative_to(run_dir.resolve())
        except ValueError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"SQLite database is outside the run directory: {source}") from exc
        destination = staging / relative
        _sqlite_copy(source, destination)
        if not _sqlite_ok(destination):
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"SQLite snapshot integrity check failed: {source}")
        sqlite_files.append(str(relative))
    if not sqlite_files:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("no SQLite database was available for the snapshot")
    if "checkpoint.json" not in copied or "run_manifest.json" not in copied:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("snapshot requires checkpoint.json and run_manifest.json")
    metadata = {
        "schema": SCHEMA,
        "completed_steps": int(completed_steps),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_signature": signature,
        "checkpoint_sha256": _sha(staging / "checkpoint.json"),
        "manifest_sha256": _sha(staging / "run_manifest.json"),
        "stream_sha256": _sha(staging / "stream.jsonl") if (staging / "stream.jsonl").exists() else None,
        "files": copied,
        "sqlite_files": sqlite_files,
        "index_base": index_relative,
        "index_files": index_files,
        "index_digest": index_digest,
    }
    _write(staging / "snapshot.json", metadata)
    _replace(staging, backup_dir / name)
    _prune(backup_dir)
    return metadata


def _prune(backup_dir: Path) -> None:
    snapshots = sorted(item for item in backup_dir.iterdir()
                       if item.is_dir() and item.name.startswith("step-"))
    for old in snapshots[:-KEEP]:
        shutil.rmtree(old, ignore_errors=True)


def _validate(path: Path, backup_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        meta = _read(path / "snapshot.json")
        if meta.get("schema") != SCHEMA:
            return None
        checkpoint = _read(path / "checkpoint.json")
        manifest = _read(path / "run_manifest.json")
        if int(checkpoint.get("completed_steps", -1)) != int(meta["completed_steps"]):
            return None
        if checkpoint.get("in_flight_step") is not None:
            return None
        signature = str(checkpoint.get("signature") or "")
        if not signature or signature != str(meta.get("run_signature") or ""):
            return None
        if str(manifest.get("signature") or "") != signature:
            return None
        stream = path / "stream.jsonl"
        rows = len([line for line in stream.read_text(encoding="utf-8").splitlines()
                    if line.strip()]) if stream.exists() else 0
        if rows != int(meta["completed_steps"]):
            return None
        if _sha(path / "checkpoint.json") != meta["checkpoint_sha256"]:
            return None
        if _sha(path / "run_manifest.json") != meta.get("manifest_sha256"):
            return None
        if meta.get("stream_sha256") and _sha(stream) != meta["stream_sha256"]:
            return None
        index_manifest = _read(backup_dir / "index_manifest.json")
        index_signature = str(index_manifest.get("run_signature") or "")
        if index_signature and index_signature != signature:
            return None
        for relative in meta.get("sqlite_files", []):
            if not _sqlite_ok(path / relative):
                return None
        index_dir = backup_dir / "index"
        actual_files = sorted(item.name for item in index_dir.iterdir() if item.is_file())
        if "index_files" in meta and sorted(meta["index_files"]) != actual_files:
            return None
        if _index_digest(index_dir) != meta.get("index_digest"):
            return None
        return meta
    except Exception:
        return None


def _stream_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()])


def _active_is_safe(run_dir: Path, minimum_steps: int) -> bool:
    """True only when the active checkpoint/stream are a consistent newer run."""
    checkpoint_path = run_dir / "checkpoint.json"
    try:
        active = _read(checkpoint_path)
        completed = int(active.get("completed_steps", -1))
        if active.get("in_flight_step") is not None or completed < minimum_steps:
            return False
        return _stream_rows(run_dir / "stream.jsonl") == completed
    except Exception:
        return False


def _active_signatures(run_dir: Path) -> set[str]:
    signatures: set[str] = set()
    for name in ("checkpoint.json", "run_manifest.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            value = str(_read(path).get("signature") or "")
        except Exception:
            continue
        if value:
            signatures.add(value)
    return signatures


def _install_file(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)


def restore_latest_snapshot(run_dir: Path, backup_dir: Path,
                            expected_signature: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Restore the newest valid snapshot if the active run is missing/unsafe.

    Restoration is staged and the checkpoint is installed last.  Until that final
    rename, the active directory still exposes its previous checkpoint, so a
    runtime reset during restore causes another restore attempt rather than a
    mixed trust state.  A snapshot from a different experiment signature is
    never installed over an active run.
    """
    run_dir, backup_dir = Path(run_dir), Path(backup_dir)
    if (run_dir / "metrics.json").exists():
        # A completed run is evidence of a finished experiment.  Never restore
        # a checkpoint snapshot over it, even when the signature matches.
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    for leftover in run_dir.glob(".restore-*.tmp"):
        shutil.rmtree(leftover, ignore_errors=True)
    if not backup_dir.exists():
        return None
    _cleanup_temporary(backup_dir)
    active_signatures = _active_signatures(run_dir)
    if len(active_signatures) > 1:
        # checkpoint and manifest disagree: installing either snapshot would mix
        # two identities.  Leave the evidence in place for human diagnosis.
        return None
    if expected_signature and active_signatures and expected_signature not in active_signatures:
        # Never replace one experiment's active run with another experiment's
        # snapshot, even if the backup directory is shared.
        return None
    wanted_signature = expected_signature or (next(iter(active_signatures))
                                               if active_signatures else None)
    if not wanted_signature:
        # With no active identity and no explicit expectation there is no safe
        # way to decide which experiment should be restored.
        return None
    candidates = []
    for item in backup_dir.iterdir():
        if item.is_dir() and item.name.startswith("step-"):
            meta = _validate(item, backup_dir)
            if meta is None:
                continue
            if meta.get("run_signature") != wanted_signature:
                continue
            candidates.append((int(meta["completed_steps"]), meta["created_at"], item, meta))
    if not candidates:
        return None
    _, _, snapshot, meta = max(candidates, key=lambda item: (item[0], item[1]))
    if _active_is_safe(run_dir, int(meta["completed_steps"])):
        return None

    staging = run_dir / f".restore-{_now()}.tmp"
    staging.mkdir(parents=True)
    checkpoint_name = "checkpoint.json"
    try:
        staged: Dict[str, Path] = {}
        for name in meta.get("files", []):
            source = snapshot / name
            if source.is_file():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                staged[name] = target
        sqlite_staged: Dict[str, Path] = {}
        for relative in meta.get("sqlite_files", []):
            source = snapshot / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _sqlite_copy(source, target)
            if not _sqlite_ok(target):
                raise RuntimeError(f"staged SQLite restore failed integrity check: {relative}")
            sqlite_staged[relative] = target
        index_staged: Dict[str, Path] = {}
        index_base = Path(meta.get("index_base") or "index")
        index_target_dir = run_dir / index_base.parent
        index_files = meta.get("index_files") or [
            item.name for item in (backup_dir / "index").iterdir() if item.is_file()]
        for name in index_files:
            source = backup_dir / "index" / name
            target = staging / "index" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            index_staged[name] = target
        if checkpoint_name not in staged:
            raise RuntimeError("snapshot is missing checkpoint.json")
        if _sha(staged[checkpoint_name]) != meta["checkpoint_sha256"]:
            raise RuntimeError("staged checkpoint failed its integrity check")

        # The index and databases are installed before the manifest/stream.  The
        # checkpoint is the commit marker and is installed last.
        for name in list(index_target_dir.glob(f"{index_base.name}.*")):
            if name.name not in index_staged and name.is_file():
                name.unlink()
        for name, target in index_staged.items():
            _install_file(target, index_target_dir / name)
        for relative, target in sqlite_staged.items():
            destination = run_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(destination) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            _install_file(target, destination)
        for name in SMALL_FILES:
            target = staged.get(name)
            if target is not None and name != checkpoint_name:
                _install_file(target, run_dir / name)
            elif target is None and name != checkpoint_name:
                stale = run_dir / name
                if stale.exists():
                    stale.unlink()
        # A restored run is no longer mid-initialisation.
        marker = run_dir / "initialization.json"
        if marker.exists():
            marker.unlink()
        _install_file(staged[checkpoint_name], run_dir / checkpoint_name)
        return meta
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def snapshot_summary(backup_dir: Path) -> Dict[str, Any]:
    """Latest valid snapshot metadata, without changing the active run."""
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return {}
    _cleanup_temporary(backup_dir)
    valid = []
    for item in backup_dir.iterdir():
        if item.is_dir() and item.name.startswith("step-"):
            meta = _validate(item, backup_dir)
            if meta is not None:
                valid.append((int(meta["completed_steps"]), meta["created_at"], item, meta))
    if not valid:
        return {}
    best = max(valid, key=lambda item: (item[0], item[1]))
    summary = dict(best[3])
    summary["snapshot_name"] = best[2].name
    return summary
