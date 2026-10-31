"""Secure wiring for Person B's defence, and the startup health check.

The protected pipeline is the **default**.  ``PersonAPipeline.from_config``
calls :func:`attach_security` unless the caller explicitly builds an
unprotected baseline config, so the CLI and library entry points cannot quietly
run with a null trust provider and a band-only policy.

The health check exists because "the security layer is installed" is not the
same as "the security layer is doing anything".  It reports the active trust
ledger, verifier backend (and whether it degraded), persistence, queue
availability and every fallback, and it refuses to call a configuration
protected when a critical control is missing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config, TrustSettings
from .contracts import DefaultPolicy, NullTrustProvider, Policy, TrustProvider, Verifier
from .trust.ledger import TrustConfig, TrustLedger
from .trust.policy import TrustPolicy
from .trust.queue import VerificationQueue
from .trust.verifier import CorroborationVerifier
from .utils.logging import get_logger

logger = get_logger(__name__)

PROTECTED = "PROTECTED"
DEGRADED = "DEGRADED"
UNPROTECTED = "UNPROTECTED"


def build_trust_config(settings: TrustSettings) -> TrustConfig:
    """Map the YAML trust settings onto the ledger's dataclass."""
    values = settings.model_dump()
    allowed = {f.name for f in dataclasses.fields(TrustConfig)}
    return TrustConfig(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class SecurityStack:
    """The four Person B components bound to one pipeline."""

    ledger: TrustLedger
    verifier: CorroborationVerifier
    policy: TrustPolicy
    queue: VerificationQueue
    enabled: bool = True

    def drain(self, now: Optional[float] = None, max_items: Optional[int] = None) -> List[Any]:
        return self.policy.drain_verification_queue(self.verifier, now=now,
                                                    max_items=max_items)

    def close(self) -> None:
        try:
            self.queue.stop_background()
        finally:
            self.ledger.close()


def attach_security(pipeline, config: Optional[Config] = None) -> Optional[SecurityStack]:
    """Install the ledger, verifier, policy and queue on *pipeline*.

    Returns the stack, or ``None`` when ``config.security.enabled`` is false.
    The caller may pass its own components before calling this; anything already
    set is preserved (this is how an evaluation baseline swaps in a policy).
    """
    config = config or pipeline.config
    security = config.security
    if not security.enabled:
        logger.warning("security disabled by configuration: this pipeline is an "
                       "UNPROTECTED baseline and must not be reported as RAGLedger")
        return None

    ledger = getattr(pipeline, "trust_provider", None)
    if isinstance(ledger, NullTrustProvider) or ledger is None:
        ledger = TrustLedger(
            config.trust_db_path(),
            store=pipeline.store,
            config=build_trust_config(config.trust),
            clock=getattr(pipeline, "clock", None),
        )

    verifier = getattr(pipeline, "verifier", None)
    if verifier is None:
        use_nli = {"nli": True, "lexical": False, "auto": None}[security.verifier]
        verifier = CorroborationVerifier(
            llm=pipeline.generator.llm,
            max_corroboration=security.max_corroboration,
            use_nli=use_nli,
            min_refute_topical_overlap=security.min_refute_topical_overlap,
            min_refute_topic_coverage=security.min_refute_topic_coverage,
            min_refute_query_coverage=security.min_refute_query_coverage,
            support_threshold=security.support_threshold,
            refute_threshold=security.refute_threshold,
            max_source_mass=security.max_source_mass,
            counterfactual_influence=security.counterfactual_influence,
        )

    queue = VerificationQueue(max_size=security.queue_size)
    policy = getattr(pipeline, "policy", None)
    if policy is None or isinstance(policy, DefaultPolicy):
        policy = TrustPolicy(
            ledger=ledger, on_quarantine=pipeline.on_quarantine, queue=queue,
            fail_open_without_verifier=security.fail_open_without_verifier,
            use_on_neutral=(security.medium_neutral_action == "use"),
        )
    elif isinstance(policy, TrustPolicy):
        policy.queue = policy.queue or queue
    else:
        # An explicit custom policy (a baseline, for example) is preserved, and
        # the health check reports the configuration as unprotected/degraded.
        logger.warning("security: preserving explicitly supplied policy %s",
                       type(policy).__name__)

    pipeline.trust_provider = ledger
    pipeline.retriever.trust = ledger
    pipeline.verifier = verifier
    pipeline.policy = policy
    stack = SecurityStack(ledger=ledger, verifier=verifier, policy=policy, queue=queue)
    pipeline.security = stack

    if getattr(pipeline.config, "security", None) is None:   # defensive, never expected
        pipeline.config.security = security
    report = security_health(pipeline)
    logger.info("security: %s | ledger=%s | verifier=%s | queue=%s",
                report["status"], report["trust_provider"],
                report["verifier_mode"] or "none",
                "on" if report["queue_enabled"] else "off")
    for warning in report["warnings"]:
        logger.warning("security: %s", warning)
    return stack


def _component_name(component: Any) -> Optional[str]:
    return type(component).__name__ if component is not None else None


def _probe_embedding(pipeline) -> Dict[str, Any]:
    """Encode a tiny probe with the *already constructed* embedder.

    This performs no download and no service startup: if the model failed to
    load, pipeline construction has already failed and the CLI reports the
    exception.  A loaded-but-broken backend is caught here.
    """
    embedder = getattr(pipeline, "embedder", None)
    name = getattr(embedder, "name", None)
    model = getattr(embedder, "model_name", None)
    detail: Dict[str, Any] = {"backend": name, "model": model}
    if embedder is None or not callable(getattr(embedder, "encode_queries", None)):
        return {**detail, "status": "failed", "detail": "no embedding backend attached"}
    try:
        matrix = embedder.encode_queries(["readiness probe"])
        shape = tuple(getattr(matrix, "shape", ()))
        dim = int(getattr(embedder, "dim", 0) or 0)
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError(f"probe returned shape {shape}")
        if dim and int(shape[1]) != dim:
            raise ValueError(f"probe returned dim {shape[1]}, expected {dim}")
        detail.update({"status": "ready", "probe_shape": list(shape), "dim": dim})
        if name == "hashing":
            detail["status"] = "fallback"
            detail["detail"] = "hashing embedder is an offline test backend"
        return detail
    except Exception as exc:  # noqa: BLE001 - readiness must report, not crash
        return {**detail, "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}"}


def _http_json(url: str, headers: Optional[Dict[str, str]] = None,
               timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _probe_generation(pipeline) -> Dict[str, Any]:
    """Check the configured generator without starting it or generating text.

    ``stub`` is a documented offline fallback.  vLLM/OpenAI-compatible and
    Ollama endpoints must answer a metadata request and list the configured
    model.  An in-process transformers model must have its tokenizer and model
    objects loaded.  No probe launches a server or downloads weights.
    """
    llm = getattr(getattr(pipeline, "generator", None), "llm", None)
    name = getattr(llm, "name", None)
    backend = getattr(getattr(pipeline, "config", None), "generation", None)
    backend_name = getattr(backend, "backend", None)
    detail: Dict[str, Any] = {"backend": backend_name, "model": name}
    if llm is None or not callable(getattr(llm, "generate", None)):
        return {**detail, "status": "failed", "detail": "no generation backend attached"}
    if name == "stub":
        return {**detail, "status": "fallback",
                "detail": "stub LLM is an offline test backend"}
    try:
        if backend_name == "openai" or hasattr(llm, "base_url") and "v1" in str(getattr(llm, "base_url", "")):
            headers = {"Content-Type": "application/json"}
            api_key = getattr(llm, "api_key", None)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            body = _http_json(f"{str(getattr(llm, 'base_url', '')).rstrip('/')}/models",
                              headers=headers)
            ids = [str(item.get("id", "")) for item in (body.get("data") or [])
                   if isinstance(item, dict)]
            if not ids:
                raise ValueError("endpoint returned no model ids")
            if name not in ids:
                raise ValueError(f"configured model {name!r} not served (available: {ids[:5]})")
            return {**detail, "status": "ready", "available_models": ids}
        if backend_name == "ollama" or hasattr(llm, "base_url"):
            base = str(getattr(llm, "base_url", "")).rstrip("/")
            body = _http_json(f"{base}/api/tags")
            models = [str(item.get("name", "")) for item in (body.get("models") or [])
                      if isinstance(item, dict)]
            if not any(item == name or item.split(":")[0] == str(name).split(":")[0]
                       for item in models):
                raise ValueError(f"configured model {name!r} not present "
                                 f"(available: {models[:5]})")
            return {**detail, "status": "ready", "available_models": models}
        if backend_name == "huggingface":
            tokenizer = getattr(llm, "tokenizer", None)
            model = getattr(llm, "model", None)
            if tokenizer is None or model is None:
                raise ValueError("transformers tokenizer/model is not loaded")
            tokenizer("readiness probe", return_tensors=None)
            return {**detail, "status": "ready",
                    "device": getattr(llm, "device", None)}
        return {**detail, "status": "unverified",
                "detail": "generation backend has no readiness probe"}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return {**detail, "status": "failed",
                "detail": f"endpoint unavailable: {type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {**detail, "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}"}


def _probe_nli(verifier, requested: Optional[str]) -> Dict[str, Any]:
    """Report whether the NLI path is actually loaded and usable."""
    mode = getattr(verifier, "mode", None)
    nli_ready = bool(getattr(verifier, "nli_ready", False))
    detail: Dict[str, Any] = {"requested": requested, "mode": mode,
                              "nli_loaded": nli_ready}
    if mode == "nli" and nli_ready:
        return {**detail, "status": "ready"}
    if mode == "lexical":
        error = getattr(verifier, "nli_error", None)
        message = "lexical fallback active"
        if requested == "nli":
            message += "; configured NLI backend did not load"
            if error:
                message += f" ({error})"
        return {**detail, "status": "fallback" if requested != "nli" else "failed",
                "detail": message}
    return {**detail, "status": "failed",
            "detail": "verifier did not report an NLI or lexical mode"}


def _probe_ledger(ledger) -> Dict[str, Any]:
    if not isinstance(ledger, TrustLedger):
        return {"status": "failed", "detail": "trust ledger is not a TrustLedger"}
    persistent = getattr(ledger, "path", ":memory:") != ":memory:"
    detail: Dict[str, Any] = {"persistent": persistent, "path": getattr(ledger, "path", None)}
    try:
        ledger._conn.execute("SELECT 1").fetchone()  # noqa: SLF001 - liveness, not data
    except Exception as exc:  # noqa: BLE001
        return {**detail, "status": "failed",
                "detail": f"ledger query failed: {type(exc).__name__}: {exc}"}
    if not persistent:
        return {**detail, "status": "fallback",
                "detail": "in-memory ledger does not survive restart"}
    return {**detail, "status": "ready"}


def _probe_queue(queue) -> Dict[str, Any]:
    if queue is None:
        return {"status": "failed", "detail": "verification queue is not configured"}
    if int(getattr(queue, "max_size", 0) or 0) <= 0:
        return {"status": "failed", "detail": "verification queue has zero capacity"}
    try:
        len(queue)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed",
                "detail": f"queue is not readable: {type(exc).__name__}: {exc}"}
    return {"status": "ready", "max_size": int(queue.max_size),
            "pending": len(queue), "dropped": int(getattr(queue, "dropped", 0))}


def security_health(pipeline) -> Dict[str, Any]:
    """Report which controls are active, and which degraded.

    ``status`` values:

    * ``PROTECTED`` - trust ledger, verifier, policy and queue are all active
      and the verifier runs the configured semantic backend.
    * ``DEGRADED``  - the controls are active but at least one runs a
      documented fallback (lexical verifier instead of NLI, in-memory ledger,
      missing queue).
    * ``UNPROTECTED`` - security is disabled or a critical component is missing.
    """
    config = getattr(pipeline, "config", None)
    security = getattr(config, "security", None)
    enabled = bool(getattr(security, "enabled", False))
    ledger = getattr(pipeline, "trust_provider", None)
    verifier = getattr(pipeline, "verifier", None)
    policy = getattr(pipeline, "policy", None)
    stack = getattr(pipeline, "security", None)
    queue = getattr(stack, "queue", None) if stack is not None else getattr(policy, "queue", None)

    ledger_active = isinstance(ledger, TrustLedger)
    verifier_active = verifier is not None
    policy_active = isinstance(policy, TrustPolicy)
    queue_active = queue is not None
    verifier_mode = getattr(verifier, "mode", None)
    persistent = bool(ledger_active and getattr(ledger, "path", ":memory:") != ":memory:")

    embedder = getattr(pipeline, "embedder", None)
    embedder_name = getattr(embedder, "name", getattr(embedder, "model_name", "unknown"))
    llm = getattr(getattr(pipeline, "generator", None), "llm", None)
    llm_name = getattr(llm, "name", "unknown")

    # Configuration validation is separate from readiness.  The fields above
    # say what is attached.  The probes below exercise the attached objects; a
    # configured name alone can never produce PROTECTED.
    requested_verifier = getattr(security, "verifier", None)
    checks = {
        "embedding": _probe_embedding(pipeline),
        "generation": _probe_generation(pipeline),
        "nli_verifier": _probe_nli(verifier, requested_verifier),
        "trust_ledger": _probe_ledger(ledger),
        "queue": _probe_queue(queue),
    }

    warnings: List[str] = []
    fallbacks: List[str] = []
    failures: List[str] = []
    if not enabled:
        warnings.append("security.enabled is false: no trust, verification or quarantine")
    if not ledger_active:
        warnings.append("trust ledger is not a TrustLedger")
    if not policy_active:
        warnings.append("policy is not TrustPolicy")
    if not queue_active:
        warnings.append("verification queue is not configured; HIGH-band items are dropped")
    for component, check in checks.items():
        state = check.get("status")
        detail = check.get("detail")
        if state == "failed":
            failures.append(f"{component}: {detail or 'readiness probe failed'}")
            warnings.append(f"{component} readiness failed: {detail or 'unknown error'}")
        elif state in {"fallback", "unverified"}:
            fallbacks.append(f"{component}: {detail or state}")
    if verifier_active and verifier_mode == "lexical" and requested_verifier != "nli":
        warnings.append("verification uses lexical overlap, not NLI; not comparable with NLI runs")
    if verifier_active and verifier_mode is None:
        fallbacks.append("verifier backend did not report a mode")

    critical = enabled and ledger_active and verifier_active and policy_active
    readiness_ready = all(check.get("status") == "ready"
                          for check in checks.values())
    critical_failure = any(check.get("status") == "failed"
                           for check in (checks["trust_ledger"], checks["queue"]))
    if not critical or critical_failure:
        status = UNPROTECTED
    elif readiness_ready:
        status = PROTECTED
    else:
        status = DEGRADED

    offline = [item for item in (embedder_name, llm_name) if item in ("hashing", "stub")]
    return {
        "status": status,
        "configuration_valid": enabled and ledger_active and verifier_active
                               and policy_active and queue_active,
        "readiness_probes": checks,
        "readiness_ready": readiness_ready,
        "readiness_failures": failures,
        "probe_policy": ("probes use already-constructed components; no model downloads "
                         "and no services are started"),
        "security_enabled": enabled,
        "trust_provider": _component_name(ledger),
        "trust_ledger_active": ledger_active,
        "trust_persistent": persistent,
        "trust_db": getattr(ledger, "path", None) if ledger_active else None,
        "verifier": _component_name(verifier),
        "verifier_active": verifier_active,
        "verifier_mode": verifier_mode,
        "verifier_degraded": bool(verifier_active and verifier_mode != "nli"),
        "policy": _component_name(policy),
        "policy_active": policy_active,
        "queue_enabled": queue_active,
        "queue_size": getattr(queue, "max_size", 0) if queue_active else 0,
        "queue_pending": len(queue) if queue_active else 0,
        "embedder": embedder_name,
        "embedding_dim": getattr(embedder, "dim", None),
        "llm": llm_name,
        "smoke_test_backends": bool(offline),
        "research_ready": bool(readiness_ready and failures == []),
        "fallbacks": fallbacks,
        "warnings": warnings,
    }


def health_line(report: Dict[str, Any]) -> str:
    """One-line startup summary (never claims PROTECTED when controls are off)."""
    return (
        f"security={report['status']} "
        f"ledger={report['trust_provider']} "
        f"verifier={report['verifier']}({report['verifier_mode'] or 'none'}) "
        f"policy={report['policy']} "
        f"queue={'on' if report['queue_enabled'] else 'off'} "
        f"embedder={report['embedder']} llm={report['llm']}"
    )
