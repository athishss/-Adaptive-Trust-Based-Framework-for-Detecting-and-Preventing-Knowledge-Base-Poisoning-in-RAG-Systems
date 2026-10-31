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


def _replace(staging: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)


def backup_snapshot(run_dir: Path, backup_dir: Path, completed_steps: int,
                    sqlite_paths: Sequence[Path], index_base: str,
                    copy_index: bool = False) -> Dict[str, Any]:
    """Snapshot a completed-step boundary atomically.

    SQLite files are copied with the SQLite backup API.  The checkpoint and
    stream must already describe exactly ``completed_steps``.
    """
    run_dir, backup_dir = Path(run_dir), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _read(run_dir / "checkpoint.json")
    if int(checkpoint.get("completed_steps", -1)) != int(completed_steps):
        raise ValueError("checkpoint completed_steps differs from the snapshot step")
    if checkpoint.get("in_flight_step") is not None:
        raise ValueError("refusing to snapshot in-flight work")
    stream = run_dir / "stream.jsonl"
    rows = len([line for line in stream.read_text(encoding="utf-8").splitlines()
                if line.strip()]) if stream.exists() else 0
    if rows != int(completed_steps):
        raise ValueError(f"stream has {rows} rows but checkpoint says {completed_steps}")
    index_dir = backup_dir / "index"
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
    index_digest = _index_digest(index_dir)
    _write(backup_dir / "index_manifest.json",
           {"digest": index_digest,
            "files": sorted(item.name for item in index_dir.iterdir() if item.is_file())})
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
        relative = source.relative_to(run_dir)
        destination = staging / relative
        _sqlite_copy(source, destination)
        if not _sqlite_ok(destination):
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"SQLite snapshot integrity check failed: {source}")
        sqlite_files.append(str(relative))
    if not sqlite_files:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("no SQLite database was available for the snapshot")
    metadata = {
        "schema": SCHEMA,
        "completed_steps": int(completed_steps),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": _sha(staging / "checkpoint.json"),
        "stream_sha256": _sha(staging / "stream.jsonl") if (staging / "stream.jsonl").exists() else None,
        "files": copied,
        "sqlite_files": sqlite_files,
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
        if int(checkpoint.get("completed_steps", -1)) != int(meta["completed_steps"]):
            return None
        if checkpoint.get("in_flight_step") is not None:
            return None
        stream = path / "stream.jsonl"
        rows = len([line for line in stream.read_text(encoding="utf-8").splitlines()
                    if line.strip()]) if stream.exists() else 0
        if rows != int(meta["completed_steps"]):
            return None
        if _sha(path / "checkpoint.json") != meta["checkpoint_sha256"]:
            return None
        if meta.get("stream_sha256") and _sha(stream) != meta["stream_sha256"]:
            return None
        for relative in meta.get("sqlite_files", []):
            if not _sqlite_ok(path / relative):
                return None
        if _index_digest(backup_dir / "index") != meta.get("index_digest"):
            return None
        return meta
    except Exception:
        return None


def restore_latest_snapshot(run_dir: Path, backup_dir: Path) -> Optional[Dict[str, Any]]:
    """Restore the newest valid snapshot if the active run is missing/unsafe."""
    run_dir, backup_dir = Path(run_dir), Path(backup_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not backup_dir.exists():
        return None
    candidates = []
    for item in backup_dir.iterdir():
        if item.is_dir() and item.name.startswith("step-"):
            meta = _validate(item, backup_dir)
            if meta is not None:
                candidates.append((int(meta["completed_steps"]), meta["created_at"], item, meta))
    if not candidates:
        return None
    _, _, snapshot, meta = max(candidates, key=lambda item: (item[0], item[1]))
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            active = _read(checkpoint_path)
            if (active.get("in_flight_step") is None and
                    int(active.get("completed_steps", -1)) >= int(meta["completed_steps"])):
                return None
        except Exception:
            pass
    for source in (backup_dir / "index").iterdir():
        if source.is_file():
            shutil.copy2(source, run_dir / source.name)
    for name in meta.get("files", []):
        shutil.copy2(snapshot / name, run_dir / name)
    for relative in meta.get("sqlite_files", []):
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(destination) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        shutil.copy2(snapshot / relative, destination)
    return meta


def snapshot_summary(backup_dir: Path) -> Dict[str, Any]:
    """Latest valid snapshot metadata, without changing active files."""
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return {}
    valid = []
    for item in backup_dir.iterdir():
        if item.is_dir() and item.name.startswith("step-"):
            meta = _validate(item, backup_dir)
            if meta is not None:
                valid.append((int(meta["completed_steps"]), meta["created_at"], meta))
    return max(valid, key=lambda item: (item[0], item[1]))[2] if valid else {}
