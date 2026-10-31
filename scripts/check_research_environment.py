#!/usr/bin/env python3
"""Preflight the environment for a real-model RAGLedger experiment.

This script is deliberately conservative and non-invasive:

* it never downloads model weights, datasets, or packages;
* it never starts vLLM, Ollama, or another service;
* it inspects configuration, installed packages, the Hugging Face *cache*,
  endpoints that are already running, disk space, and the persistent ledger.

A model name in YAML is not evidence that the model is available.  Static
checks alone can therefore only report DEGRADED.  Pass ``--functional`` to run
the existing ``doctor`` probes against already-loaded components (with offline
Hugging Face variables set); only a PROTECTED doctor result can produce READY.

Exit codes: 0 READY, 1 DEGRADED, 2 BLOCKED.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trace_rag.config import Config  # noqa: E402

READY = "READY"
DEGRADED = "DEGRADED"
BLOCKED = "BLOCKED"

REQUIRED_MODULES = {
    "torch": ("torch", "pip install torch  # use the CUDA build from pytorch.org"),
    "transformers": ("transformers", "pip install transformers"),
    "faiss": ("faiss-cpu", "pip install faiss-cpu  # or a CUDA-enabled faiss build"),
    "pyarrow": ("pyarrow", "pip install pyarrow"),
}

PREFLIGHT_STORAGE_SCHEMA = "ragledger.preflight.storage.v1"
STORAGE_PATH_FIELDS = ("provenance_db", "answer_db", "index_path",
                       "scorer_path", "trust_db")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _safe_storage_name(value: str) -> str:
    name = Path(value).name
    if not name or name in {".", ".."}:
        raise ValueError(f"cannot derive a safe preflight filename from {value!r}")
    return name


def isolate_storage_config(config: Config, root: Path) -> Tuple[Config, List[str]]:
    """Copy *config* with every storage path confined to ``root``.

    The functional doctor is a real pipeline construction and therefore writes
    SQLite files, and (when it loads an index) index files.  It must never write
    those into the experiment's storage root: the runner deliberately refuses a
    non-empty output directory that has no checkpoint.  Model, security and
    generation settings are copied unchanged; only storage locations move.

    Absolute paths and relative paths containing ``..`` cannot be confined by
    changing ``storage.root`` alone, so they are rebased to their basename under
    the probe root.  The returned list names those fields for the report.
    """
    probe = config.model_copy(deep=True)
    probe.storage.root = str(root)
    rebased: List[str] = []

    for field in STORAGE_PATH_FIELDS:
        value = getattr(probe.storage, field)
        if value == ":memory:":
            continue
        candidate = Path(value)
        if candidate.is_absolute() or not _is_within(root / candidate, root):
            setattr(probe.storage, field, _safe_storage_name(value))
            rebased.append(f"storage.{field}")

    configured_trust = probe.security.trust_db
    if configured_trust and configured_trust != ":memory:":
        candidate = Path(configured_trust)
        if candidate.is_absolute() or not _is_within(root / candidate, root):
            probe.security.trust_db = _safe_storage_name(configured_trust)
            rebased.append("security.trust_db")

    effective = {
        "storage.root": Path(probe.storage.root),
        **{f"storage.{field}": probe.resolve_path(getattr(probe.storage, field))
           for field in STORAGE_PATH_FIELDS},
        "trust_ledger": Path(probe.trust_db_path(create_parent=False))
        if probe.trust_db_path(create_parent=False) != ":memory:" else None,
    }
    escaped = [name for name, path in effective.items()
               if path is not None and not _is_within(path, root)]
    if escaped:
        raise ValueError("preflight storage isolation failed for: " + ", ".join(escaped))
    return probe, rebased


def _probe_storage_root(parent: Optional[Path]) -> Tuple[Path, Optional[Path]]:
    """Create a unique probe directory and return (root, explicit parent)."""
    if parent is None:
        return Path(tempfile.mkdtemp(prefix="ragledger-preflight-storage-")), None
    parent = Path(parent).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="probe-", dir=parent)), parent


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _check(name: str, status: str, detail: str, remediation: str = "") -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail,
            "remediation": remediation}


def _memory_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:  # noqa: BLE001 - diagnostics must not crash
            return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def _human_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def system_info() -> Dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "memory_human": _human_bytes(_memory_bytes()),
        "disk_free_bytes": usage.free,
        "disk_free_human": _human_bytes(usage.free),
    }


def torch_info() -> Dict[str, Any]:
    if not _module_available("torch"):
        return {"available": False, "version": None, "cuda": False,
                "device_count": 0, "devices": []}
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": None, "cuda": False,
                "device_count": 0, "devices": [],
                "error": f"{type(exc).__name__}: {exc}"}
    cuda = bool(torch.cuda.is_available())
    devices: List[Dict[str, Any]] = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free = total = None
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                free, total = int(free_bytes), int(total_bytes)
            except Exception:  # noqa: BLE001 - older drivers
                total = int(getattr(props, "total_memory", 0)) or None
            devices.append({
                "index": index,
                "name": props.name,
                "total_vram_bytes": total,
                "free_vram_bytes": free,
                "total_vram_human": _human_bytes(total),
                "free_vram_human": _human_bytes(free),
            })
    return {
        "available": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda": cuda,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": len(devices),
        "devices": devices,
    }


def _model_is_local(model: str) -> bool:
    return Path(model).expanduser().exists()


def model_cache_status(model: str, revision: Optional[str] = None) -> Dict[str, Any]:
    """Check the local Hugging Face cache without downloading anything."""
    if _model_is_local(model):
        return {"model": model, "cached": True, "source": "local_path"}
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        cached = try_to_load_from_cache(model, "config.json", revision=revision)
        if cached is not None and Path(str(cached)).exists():
            return {"model": model, "cached": True, "source": "huggingface_hub"}
        return {"model": model, "cached": False, "source": "huggingface_hub"}
    except ImportError:
        pass
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder = root / ("models--" + model.replace("/", "--"))
    pattern = f"snapshots/{revision}/config.json" if revision else "snapshots/*/config.json"
    snapshots = list(folder.glob(pattern))
    return {"model": model, "cached": bool(snapshots),
            "source": "cache_scan" if snapshots else "cache_scan_missing"}


def _http_json(url: str, headers: Optional[Dict[str, str]], timeout: float) -> Any:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def endpoint_status(config: Config) -> Dict[str, Any]:
    generation = config.generation
    backend = generation.backend
    if backend in {"stub", "huggingface"}:
        return {"status": "not_applicable", "backend": backend,
                "detail": "no HTTP endpoint is used by this backend"}
    base_url = generation.base_url.rstrip("/")
    timeout = min(float(config.security.probe_timeout_s or 10.0), 10.0)
    headers: Dict[str, str] = {}
    if backend == "openai":
        key = os.environ.get(generation.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = f"{base_url}/models"
        key_name = "data"
    elif backend == "ollama":
        url = f"{base_url}/api/tags"
        key_name = "models"
    else:
        return {"status": "failed", "backend": backend,
                "detail": f"unsupported generation backend {backend!r}"}
    try:
        body = _http_json(url, headers, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            ValueError) as exc:
        # Never include request headers or an API key in this message.
        return {"status": "failed", "backend": backend, "base_url": base_url,
                "detail": f"endpoint unavailable: {type(exc).__name__}: {exc}"}
    if not isinstance(body, dict):
        return {"status": "failed", "backend": backend, "base_url": base_url,
                "detail": "endpoint returned a non-object response"}
    entries = body.get(key_name) or []
    if backend == "openai":
        available = [str(item.get("id", "")) for item in entries if isinstance(item, dict)]
    else:
        available = [str(item.get("name", "")) for item in entries if isinstance(item, dict)]
    wanted = generation.model_name
    served = any(item == wanted or
                 (backend == "ollama" and item.split(":")[0] == wanted.split(":")[0])
                 for item in available)
    return {
        "status": "ready" if served else "failed",
        "backend": backend,
        "base_url": base_url,
        "configured_model": wanted,
        "available_models": available[:20],
        "model_served": served,
        "detail": ("configured model is served" if served else
                   f"configured model {wanted!r} is not served"),
    }


def ledger_status(config: Config) -> Dict[str, Any]:
    # Never use Config.path here: it creates the parent directory, and a read-only
    # preflight must not leave an empty experiment tree behind.
    path = config.trust_db_path(create_parent=False)
    if path == ":memory:":
        return {"status": "failed", "path": path,
                "detail": "trust ledger is in-memory; research runs require persistence",
                "remediation": "set storage.trust_db (or security.trust_db) to a file path"}
    ledger = Path(path)
    if not ledger.exists():
        parent = ledger.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        parent_writable = parent.exists() and os.access(parent, os.W_OK)
        return {"status": "ready" if parent_writable else "failed",
                "path": str(ledger), "initialized": False,
                "detail": ("path is writable; the run will create it" if parent_writable
                           else "parent directory is not writable"),
                "remediation": "" if parent_writable else "choose a writable storage.root"}
    try:
        connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=2.0)
        try:
            connection.execute("SELECT 1").fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {"status": "failed", "path": str(ledger), "initialized": True,
                "detail": f"cannot open SQLite ledger: {type(exc).__name__}: {exc}",
                "remediation": "remove or repair the corrupt trust database"}
    return {"status": "ready", "path": str(ledger), "initialized": True,
            "detail": "persistent SQLite ledger opens successfully"}


def dataset_status(corpus: Optional[str], queries: Optional[str],
                   qrels: Optional[str]) -> Dict[str, Any]:
    if not any((corpus, queries, qrels)):
        return {"status": "not_checked",
                "detail": "pass --corpus/--queries/--qrels to check the prepared subset"}
    missing = [name for name, value in
               (("corpus", corpus), ("queries", queries), ("qrels", qrels))
               if value and not Path(value).exists()]
    if missing:
        return {"status": "failed", "missing": missing,
                "detail": f"missing dataset files: {', '.join(missing)}",
                "remediation": "python scripts/download_data.py --dataset nq "
                               "--out data/nq --subset 200000 --queries 500"}
    sizes = {name: Path(value).stat().st_size for name, value in
             (("corpus", corpus), ("queries", queries), ("qrels", qrels)) if value}
    return {"status": "ready", "sizes": sizes, "detail": "dataset files exist"}


def functional_doctor(config: Config, timeout: float,
                      preflight_storage_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run the existing doctor probes with offline HF env vars.

    The offline flags prevent an attempted model download.  If weights are not
    cached, the doctor fails fast and that failure is reported honestly.

    The doctor constructs a real :class:`PersonAPipeline`, which creates SQLite
    files.  It therefore always runs with a storage root isolated from the
    experiment, and that temporary tree is removed afterwards.  ``config`` is
    not mutated, so the report can still show the intended experiment paths.
    """
    probe_root, explicit_parent = _probe_storage_root(preflight_storage_root)
    isolation: Dict[str, Any] = {
        "schema": PREFLIGHT_STORAGE_SCHEMA,
        "intended_root": str(config.storage.root),
        "probe_root": str(probe_root),
        "explicit_parent": str(explicit_parent) if explicit_parent else None,
        "rebased_paths": [],
        "cleaned": False,
    }
    result: Dict[str, Any]
    try:
        probe_config, rebased = isolate_storage_config(config, probe_root)
        isolation["rebased_paths"] = rebased
        # Doctor must receive the already-resolved config, including every --set
        # override.  Checking a YAML file and running a different command was the
        # exact mismatch this preflight is meant to prevent.
        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        })
        with tempfile.TemporaryDirectory(prefix="ragledger-preflight-") as directory:
            resolved_path = Path(directory) / "resolved.yaml"
            probe_config.dump(resolved_path)
            command = [sys.executable, "-m", "trace_rag.cli",
                       "--config", str(resolved_path), "doctor"]
            try:
                completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                           text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                result = {"status": "failed",
                          "detail": f"doctor timed out after {timeout:.0f}s"}
            else:
                output = completed.stdout.strip()
                payload: Optional[Dict[str, Any]] = None
                start = output.find("{")
                if start >= 0:
                    try:
                        payload = json.loads(output[start:])
                    except json.JSONDecodeError:
                        payload = None
                if payload is None:
                    detail = (completed.stderr or output or "doctor returned no JSON").strip()
                    result = {"status": "failed", "exit_code": completed.returncode,
                              "detail": detail[-1000:]}
                else:
                    result = {
                        "status": payload.get("status", "UNKNOWN"),
                        "exit_code": completed.returncode,
                        "security": payload,
                        "detail": payload.get("status_reason", ""),
                    }
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)
        isolation["cleaned"] = not probe_root.exists()
        if explicit_parent is not None:
            # Do not remove a caller-owned preflight parent; only assert that the
            # unique child tree is gone.  An empty parent is intentional.
            isolation["parent_empty"] = not any(explicit_parent.iterdir())
    result["storage_isolation"] = isolation
    return result


def classify_status(checks: List[Dict[str, str]],
                    doctor: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Turn static checks and the optional functional probe into one verdict."""
    failures = [item for item in checks if item["status"] == "fail"]
    if failures:
        return BLOCKED, f"{len(failures)} blocking check(s) failed"
    if doctor is None:
        return DEGRADED, "static checks passed, but no functional model probes were run"
    report = doctor.get("status")
    if report == "PROTECTED":
        return READY, "static checks and functional security doctor passed"
    if report == "DEGRADED":
        return DEGRADED, "static checks passed, but doctor is DEGRADED"
    return BLOCKED, f"functional security doctor failed: {report or 'unknown status'}"


def resolve_config(config_path: Path, overrides: Optional[List[str]] = None) -> Config:
    """Load the exact effective configuration shared with the runner."""
    from trace_rag.cli import apply_overrides

    return apply_overrides(Config.load(config_path), overrides)


def build_report(config_path: Path, functional: bool, functional_timeout: float,
                 corpus: Optional[str], queries: Optional[str],
                 qrels: Optional[str], overrides: Optional[List[str]] = None,
                 preflight_storage_root: Optional[Path] = None) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []
    info = system_info()
    checks.append(_check("python", "ok" if sys.version_info >= (3, 10) else "fail",
                         f"{info['python']} on {info['platform']}",
                         "Python 3.10+ is required; 3.11+ is recommended"))
    checks.append(_check("disk", "ok" if info["disk_free_bytes"] >= 20 * 1024**3 else "warn",
                         f"{info['disk_free_human']} free at {ROOT}",
                         "free at least ~20 GiB for a 200K subset and model cache"))
    for module, (package, remediation) in REQUIRED_MODULES.items():
        available = _module_available(module)
        checks.append(_check(
            f"dependency:{module}", "ok" if available else "fail",
            ("installed " + _version(package)) if available else "not installed",
            remediation,
        ))

    torch_data = torch_info()
    if not torch_data["available"]:
        checks.append(_check("gpu", "fail", "PyTorch is unavailable, so CUDA cannot be checked",
                             "install the CUDA build of PyTorch"))
    elif not torch_data["cuda"]:
        checks.append(_check("gpu", "fail",
                             f"CUDA unavailable (torch {torch_data['version']}); "
                             "Llama-3.1-8B on CPU is not a practical research configuration",
                             "run on a CUDA GPU; for a tiny integration pilot use a smaller "
                             "served model and set generation.model_name explicitly"))
    else:
        first = torch_data["devices"][0]
        checks.append(_check("gpu", "ok",
                             f"{first['name']} with {first['total_vram_human']} total VRAM "
                             f"({first['free_vram_human']} free); torch {torch_data['version']}, "
                             f"CUDA {torch_data['cuda_version']}"))

    try:
        config = resolve_config(config_path, overrides)
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("config", "fail",
                             f"cannot load {config_path}: {type(exc).__name__}: {exc}",
                             "fix the YAML or pass --config"))
        return {"status": BLOCKED, "checks": checks, "system": info,
                "torch": torch_data, "error": "configuration invalid"}

    checks.append(_check("config", "ok", f"loaded {config_path}",
                         "fix the YAML if validation fails"))
    if config.security.verifier != "nli":
        checks.append(_check("verifier", "fail",
                             f"security.verifier={config.security.verifier!r}; research runs "
                             "require the real NLI verifier",
                             "set security.verifier: nli"))
    else:
        checks.append(_check("verifier", "ok", "security.verifier=nli (must also pass the "
                             "functional NLI probe)", "install transformers and cache "
                             "cross-encoder/nli-deberta-v3-base"))
    if config.embedding.backend != "huggingface":
        checks.append(_check("embedding", "fail",
                             f"embedding.backend={config.embedding.backend!r}; a real run "
                             "requires huggingface embeddings",
                             "set embedding.backend: huggingface and model_name: facebook/contriever"))
    else:
        checks.append(_check("embedding", "ok",
                             f"embedding model {config.embedding.model_name} (cache below)"))
    if config.index.backend != "faiss":
        checks.append(_check("index", "fail", f"index.backend={config.index.backend!r}",
                             "set index.backend: faiss"))
    else:
        checks.append(_check("index", "ok", f"FAISS {config.index.faiss_kind}"))

    cache_checks = []
    for role, model, revision in (("embedding", config.embedding.model_name,
                                   config.embedding.revision),
                                  ("nli", config.security.nli_model,
                                   config.security.nli_revision)):
        cache = model_cache_status(model, revision)
        cache_checks.append({"role": role, **cache})
        checks.append(_check(
            f"cache:{role}", "ok" if cache["cached"] else "fail",
            f"{model}: " + ("cached" if cache["cached"] else "not found in local cache"),
            f"allow the first real run to download {model} once, or pre-populate the "
            "Hugging Face cache; the preflight itself never downloads weights",
        ))
    endpoint = endpoint_status(config)
    endpoint_level = {"ready": "ok", "not_applicable": "warn", "failed": "fail"}[
        endpoint["status"]]
    checks.append(_check("generation_endpoint", endpoint_level, endpoint["detail"],
                         "start the configured vLLM/Ollama server and serve exactly "
                         f"{config.generation.model_name}"))
    ledger = ledger_status(config)
    checks.append(_check("trust_ledger", "ok" if ledger["status"] == "ready" else "fail",
                         ledger["detail"], ledger.get("remediation", "")))
    data = dataset_status(corpus, queries, qrels)
    data_level = {"ready": "ok", "not_checked": "warn", "failed": "fail"}[data["status"]]
    checks.append(_check("dataset", data_level, data["detail"],
                         data.get("remediation", "")))

    doctor: Optional[Dict[str, Any]] = None
    if functional:
        # Even when the caller points storage.root at the experiment, the
        # functional doctor runs against a disposable tree.  This prevents a
        # successful preflight from making the experiment directory non-empty.
        doctor = functional_doctor(config, functional_timeout, preflight_storage_root)
        checks.append(_check(
            "functional_doctor",
            "ok" if doctor.get("status") == "PROTECTED" else "fail",
            f"doctor status={doctor.get('status')}: {doctor.get('detail', '')}",
            "resolve the doctor readiness failures; a real-model run requires PROTECTED",
        ))
        probes = (doctor.get("security") or {}).get("readiness_probes") or {}
        for probe_name in ("embedding", "generation", "nli_verifier", "trust_ledger"):
            probe = probes.get(probe_name) or {}
            checks.append(_check(
                f"functional:{probe_name}",
                "ok" if probe.get("status") == "ready" else "fail",
                f"{probe.get('status', 'missing')}: {probe.get('detail', '')}",
                "resolve this model/storage probe; a PROTECTED doctor is required",
            ))

    failures = [item for item in checks if item["status"] == "fail"]
    warnings = [item for item in checks if item["status"] == "warn"]
    status, reason = classify_status(checks, doctor)
    return {
        "status": status,
        "status_reason": reason,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "system": info,
        "torch": torch_data,
        "config_path": str(config_path),
        "config": {
            "seed": config.seed,
            "embedding_model": config.embedding.model_name,
            "embedding_revision": config.embedding.revision,
            "embedding_precision": config.embedding.precision,
            "embedding_device": config.embedding.device,
            "index": f"{config.index.backend}/{config.index.faiss_kind}",
            "generation_backend": config.generation.backend,
            "generation_model": config.generation.model_name,
            "generation_revision": config.generation.revision,
            "generation_precision": config.generation.precision,
            "generation_quantization": config.generation.quantization,
            "base_url": config.generation.base_url,
            "verifier": config.security.verifier,
            "nli_model": config.security.nli_model,
            "nli_revision": config.security.nli_revision,
            "trust_db": config.trust_db_path(create_parent=False),
            "storage_root": config.storage.root,
        },
        "model_cache": cache_checks,
        "generation_endpoint": endpoint,
        "trust_ledger": ledger,
        "dataset": data,
        "functional_doctor": doctor,
    }


def print_report(report: Dict[str, Any]) -> None:
    info = report["system"]
    print("RAGLedger research-environment preflight")
    print(f"  python: {info['python']}  ({info['platform']})")
    print(f"  cpu: {info['cpu_count']} logical cores   memory: {info['memory_human']}")
    print(f"  disk free: {info['disk_free_human']}")
    print()
    for check in report["checks"]:
        marker = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[check["status"]]
        print(f"  [{marker:4}] {check['name']}: {check['detail']}")
        if check["status"] == "fail" and check["remediation"]:
            print(f"         fix: {check['remediation']}")
    print()
    print(f"RESULT: {report['status']} - {report['status_reason']}")
    if report.get("failures"):
        print("Blocking checks:")
        for item in report["failures"]:
            print(f"  - {item['name']}: {item['detail']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/nq_gpu.yaml")
    parser.add_argument("--functional", action="store_true",
                        help="run the doctor's real inference probes with HF offline flags; "
                             "only this mode can report READY")
    parser.add_argument("--functional-timeout", type=float, default=900.0)
    parser.add_argument("--preflight-storage-root", default=None,
                        help="parent directory for the disposable functional-doctor storage; "
                             "the doctor never writes into the experiment's storage.root")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=None,
                        help="override the same config keys as run_trust_stream.py, e.g. "
                             "--set generation.backend=ollama --set generation.model_name=qwen2.5:7b")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = build_report(Path(args.config), args.functional, args.functional_timeout,
                          args.corpus, args.queries, args.qrels, args.set,
                          Path(args.preflight_storage_root)
                          if args.preflight_storage_root else None)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return {READY: 0, DEGRADED: 1, BLOCKED: 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
