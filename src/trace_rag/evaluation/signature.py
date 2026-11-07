"""Run signature: what a cached matrix case must match to be reused.

``resume`` exists so a Colab disconnect does not throw away hours of GPU work.
It must never let a case computed under *different* conditions leak into a new
run.  The signature therefore covers everything that can change a result:

* the code (git HEAD of the repository, plus whether the tree is dirty),
* the config file's bytes and every ``--set`` override,
* the fixture content (clean/late corpus, questions, targets, false answers),
* the target limit / repeats,
* caller-supplied extras: dataset file hashes, scorer hash, model digests, seed
  block, library versions.

Only the SHA-256 digest of the canonical JSON is stored per case; the full
signature is written to ``results.json`` so a run can be audited.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def file_sha256(path: str | Path | None) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(root: str | Path | None = None) -> Dict[str, Any]:
    cwd = str(root or Path(__file__).resolve().parent)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=20, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "config"],
                                    cwd=cwd, capture_output=True, text=True, timeout=20,
                                    check=True).stdout.strip())
        return {"head": head, "dirty": dirty}
    except Exception:  # noqa: BLE001 - not a git checkout (e.g. an installed wheel)
        return {"head": None, "dirty": None}


def library_versions() -> Dict[str, Optional[str]]:
    """Versions of every library that can change a number; ``None`` if absent."""
    from importlib import metadata
    names = ["torch", "transformers", "sentence-transformers", "faiss-cpu", "faiss-gpu",
             "numpy", "scipy", "scikit-learn", "pyarrow", "pandas", "nltk", "rouge-score",
             "vllm", "huggingface-hub", "joblib", "pydantic"]
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def fixture_digest(fixture: Any) -> str:
    payload = {name: getattr(fixture, name) for name in
               ("clean", "late_clean", "questions", "targets", "false_answers")}
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_signature(fixture: Any, *, config_path: str | Path | None,
                  overrides: Sequence[str] = (), target_limit: int = 0,
                  repeat_targets: int = 0,
                  extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "code": git_state(),
        "config_sha256": file_sha256(config_path),
        "overrides": sorted(str(o) for o in overrides),
        "fixture_sha256": fixture_digest(fixture),
        "target_limit": int(target_limit),
        "repeat_targets": int(repeat_targets),
        "extra": dict(extra or {}),
    }
    # Library versions are recorded for audit but are NOT part of the digest of a
    # smoke run's identity unless the caller opts in via ``extra``; including them
    # would make every pip upgrade silently recompute the whole matrix.
    canonical = json.dumps(body, sort_keys=True, default=str, ensure_ascii=False)
    return {**body, "libraries": library_versions(),
            "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
