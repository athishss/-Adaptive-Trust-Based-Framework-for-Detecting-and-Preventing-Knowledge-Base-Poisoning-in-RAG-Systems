"""Trust ledger with Beta-distribution trust model (plan Section 4.6 / 4.7).

Stores per-entity (doc, family, source) trust as Beta(alpha, beta) distributions.
The expected value alpha/(alpha+beta) gives the trust score.  Updates are
*asymmetric*: losing trust is faster than gaining it, which prevents an
attacker from gaming the system by sending many supportive passages to build
trust before attacking.

Person A never writes trust -- only this ledger does, based on verified
outcomes from the :class:`CorroborationVerifier`.

Three levels of trust hierarchy:
    document (chunk)  ->  family (near-duplicates)  ->  source (contributor)

The effective trust ``t_eff`` blends all three using empirical-Bayes weighting:
observations at the document level are trusted more when abundant; otherwise
the family and source levels dominate (cold-start fallback).

Key features from plan Section 4.6:
    - **Source-inherited prior**: new docs inherit alpha_0 = kappa * T_source + 1
    - **T_cap**: inherited trust capped (default 0.6) to stop slow-burn exploits
    - **Trust decay**: gamma-based forgetting per N queries
    - **Burst-aware prior**: new sources in burst windows start with lower trust
    - **Structured audit log**: every state transition is recorded with evidence

State machine (per document):
    TRUSTED -(refute)-> MONITORED -(2nd refute)-> QUARANTINED -(3rd)-> REJECTED
       ^                     ^
       +-(support, t>0.5)----+
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..contracts import TrustSnapshot, TrustStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trust_entities (
    entity_id       TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK (entity_type IN ('doc', 'family', 'source')),
    alpha           REAL NOT NULL DEFAULT 1.0,
    beta            REAL NOT NULL DEFAULT 1.0,
    status          TEXT NOT NULL DEFAULT 'TRUSTED',
    n_observations  INTEGER NOT NULL DEFAULT 0,
    n_refutations   INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    quarantined_at  REAL,
    is_burst_source INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (entity_id, entity_type)
);

CREATE TABLE IF NOT EXISTS doc_registry (
    doc_id      TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    family_id   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    old_status  TEXT NOT NULL,
    new_status  TEXT NOT NULL,
    trigger     TEXT NOT NULL,
    evidence    TEXT,
    timestamp   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trust_status ON trust_entities(entity_type, status);
CREATE INDEX IF NOT EXISTS idx_doc_source   ON doc_registry(source_id);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_id, entity_type);
"""


@dataclass
class TrustConfig:
    """Tuneable parameters for the trust model.

    Every parameter maps to a named concept in plan Section 4.6.
    Sensitivity sweeps (B5) vary each of these.
    """

    # ---- Beta update magnitudes (asymmetric: refute > support) ----
    support_doc_alpha: float = 1.0
    support_family_alpha: float = 0.5
    support_source_alpha: float = 0.3
    refute_doc_beta: float = 2.0
    refute_family_beta: float = 1.0
    refute_source_beta: float = 0.5

    # ---- Empirical-Bayes blending ----
    prior_weight: float = 5.0          # m in Section 4.6: n_d / (n_d + m)

    # ---- Source-inherited prior (Section 4.6) ----
    kappa: float = 3.0                 # prior strength: alpha_0 = kappa * T_source + 1

    # ---- T_cap: maximum inherited trust for new documents (Section 4.6) ----
    t_cap: float = 0.6                 # new docs from reputable sources capped here

    # ---- Trust decay / forgetting (Section 4.6) ----
    decay_gamma: float = 0.95          # shrinkage per decay cycle: alpha <- 1 + gamma*(alpha-1)
    decay_interval: int = 100          # apply decay every N observations system-wide

    # ---- State machine thresholds ----
    quarantine_refutations: int = 2    # refutations for MONITORED -> QUARANTINED
    reject_refutations: int = 3        # refutations for QUARANTINED -> REJECTED
    recovery_threshold: float = 0.5    # t above this: MONITORED -> TRUSTED
    quarantine_t_eff: float = 0.20     # T_eff below this -> QUARANTINED (Section 4.7)
    monitored_t_eff: float = 0.60      # T_eff below this -> MONITORED (Section 4.7)

    # ---- HIGH-band passive penalty ----
    high_band_beta_penalty: float = 0.3

    # ---- Burst-aware Sybil defence (Section 4.6 / B2) ----
    burst_trust_discount: float = 0.3  # new sources in burst: prior reduced by this factor


class TrustLedger:
    """SQLite-backed trust provider with Beta-distribution model.

    Satisfies the ``TrustProvider`` protocol from ``contracts.py``:
      * ``get_trust(doc_ids)``   — batch read, called once per query
      * ``blocked_doc_ids()``    — QUARANTINED + REJECTED, cached
    """

    def __init__(self, path: str | Path, store: Any = None,
                 config: Optional[TrustConfig] = None) -> None:
        self.path = str(path)
        self.store = store           # optional ProvenanceStore for source/family lookups
        self.config = config or TrustConfig()
        self._lock = threading.RLock()
        self._blocked_cache: Optional[Set[str]] = None
        self._global_obs_count: int = 0   # for decay scheduling

        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        except sqlite3.OperationalError as exc:
            raise sqlite3.OperationalError(
                f"cannot open the trust ledger at {self.path}: {exc}"
            ) from exc
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Migrate: add columns that may not exist in older databases
            for col, defn in [("is_burst_source", "INTEGER NOT NULL DEFAULT 0")]:
                try:
                    self._conn.execute(
                        f"ALTER TABLE trust_entities ADD COLUMN {col} {defn}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            self._conn.commit()

    # ------------------------------------------------------------------
    #  TrustProvider protocol
    # ------------------------------------------------------------------

    def get_trust(self, doc_ids: Sequence[str]) -> Dict[str, TrustSnapshot]:
        """Batch read trust snapshots for *doc_ids*.  Called once per query."""
        if not doc_ids:
            return {}
        doc_ids = list(doc_ids)
        now = time.time()

        with self._lock:
            mappings = self._get_mappings(doc_ids)
            trust_data = self._batch_get_entities(doc_ids, mappings)

        result: Dict[str, TrustSnapshot] = {}
        for doc_id in doc_ids:
            source_id, family_id = mappings.get(doc_id, ("unknown", "unknown"))

            doc_e = trust_data.get(("doc", doc_id))
            fam_e = trust_data.get(("family", family_id))
            src_e = trust_data.get(("source", source_id))

            t_doc = doc_e["trust"] if doc_e else 0.5
            t_family = fam_e["trust"] if fam_e else 0.5
            t_source = src_e["trust"] if src_e else 0.5
            n_obs = doc_e["n_observations"] if doc_e else 0
            status_str = doc_e["status"] if doc_e else "TRUSTED"

            # Empirical-Bayes blend: weight doc level by observation confidence
            w_doc = min(1.0, n_obs / max(1.0, self.config.prior_weight))
            t_eff = w_doc * t_doc + (1.0 - w_doc) * (0.5 * t_family + 0.5 * t_source)
            # Conservative clamp: a bad source cannot have trusted docs
            t_eff = min(t_eff, t_source)
            # T_cap: new docs from reputable sources cannot exceed t_cap
            # until they have enough observations to earn higher trust
            if n_obs < self.config.prior_weight:
                t_eff = min(t_eff, self.config.t_cap)
            t_eff = float(max(0.0, min(1.0, t_eff)))

            try:
                status = TrustStatus(status_str)
            except ValueError:
                status = TrustStatus.TRUSTED

            result[doc_id] = TrustSnapshot(
                doc_id=doc_id, source_id=source_id, family_id=family_id,
                t_doc=float(t_doc), t_family=float(t_family), t_source=float(t_source),
                t_eff=t_eff, status=status, n_doc_observations=int(n_obs),
                as_of=now,
            )
        return result

    def blocked_doc_ids(self) -> set:
        """Return all QUARANTINED + REJECTED doc_ids (cached until next mutation)."""
        if self._blocked_cache is not None:
            return set(self._blocked_cache)
        with self._lock:
            rows = self._conn.execute(
                "SELECT entity_id FROM trust_entities "
                "WHERE entity_type = 'doc' AND status IN ('QUARANTINED', 'REJECTED')"
            ).fetchall()
            self._blocked_cache = {row["entity_id"] for row in rows}
        return set(self._blocked_cache)

    # ------------------------------------------------------------------
    #  Trust updates (called by TrustPolicy after verification)
    # ------------------------------------------------------------------

    def record_observation(self, doc_id: str, source_id: str, family_id: str,
                           outcome: str, timestamp: Optional[float] = None) -> None:
        """Update trust based on a verification outcome.

        *outcome* must be ``'SUPPORT'``, ``'REFUTE'`` or ``'NEUTRAL'``.
        NEUTRAL observations are recorded but do not change alpha/beta.
        """
        if outcome not in ("SUPPORT", "REFUTE", "NEUTRAL"):
            raise ValueError(f"outcome must be SUPPORT/REFUTE/NEUTRAL, got {outcome!r}")
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            self._register_doc(doc_id, source_id, family_id)
            is_burst = self._is_burst_source(source_id)

            if outcome == "SUPPORT":
                self._update_entity("doc", doc_id,
                                    alpha_delta=self.config.support_doc_alpha, ts=ts,
                                    source_id=source_id, is_burst=is_burst)
                self._update_entity("family", family_id,
                                    alpha_delta=self.config.support_family_alpha, ts=ts)
                self._update_entity("source", source_id,
                                    alpha_delta=self.config.support_source_alpha, ts=ts)
            elif outcome == "REFUTE":
                self._update_entity("doc", doc_id,
                                    beta_delta=self.config.refute_doc_beta, ts=ts,
                                    is_refutation=True,
                                    source_id=source_id, is_burst=is_burst)
                self._update_entity("family", family_id,
                                    beta_delta=self.config.refute_family_beta, ts=ts,
                                    is_refutation=True)
                self._update_entity("source", source_id,
                                    beta_delta=self.config.refute_source_beta, ts=ts,
                                    is_refutation=True)
            else:
                # NEUTRAL: ensure the entity exists so n_observations is tracked
                self._ensure_entity("doc", doc_id, ts,
                                    source_id=source_id, is_burst=is_burst)
                self._conn.execute(
                    "UPDATE trust_entities SET n_observations = n_observations + 1, "
                    "updated_at = ? WHERE entity_id = ? AND entity_type = 'doc'",
                    (ts, doc_id),
                )

            # State machine transitions (doc-level only)
            if outcome == "REFUTE":
                self._check_demotion(doc_id, ts)
            elif outcome == "SUPPORT":
                self._check_recovery(doc_id, ts)

            self._conn.commit()
            self._blocked_cache = None  # invalidate

    def record_high_band(self, doc_id: str, source_id: str, family_id: str,
                         timestamp: Optional[float] = None) -> None:
        """Apply a small trust penalty for a HIGH-band assessment (no verification).

        The passage was too suspicious for even the verifier, so it gets a mild
        beta bump.  This is weaker than a full REFUTE — it only moves the
        Beta distribution slightly towards distrust.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._register_doc(doc_id, source_id, family_id)
            is_burst = self._is_burst_source(source_id)
            self._update_entity("doc", doc_id,
                                beta_delta=self.config.high_band_beta_penalty, ts=ts,
                                source_id=source_id, is_burst=is_burst)
            self._update_entity("source", source_id,
                                beta_delta=self.config.high_band_beta_penalty * 0.3, ts=ts)
            # Check if trust dropped below thresholds
            self._check_demotion(doc_id, ts, trigger="HIGH_BAND")
            self._conn.commit()
            self._blocked_cache = None

    def quarantine(self, doc_ids: Sequence[str], reason: str = "quarantined",
                   timestamp: Optional[float] = None) -> List[str]:
        """Directly quarantine specific doc_ids.  Returns the list actually transitioned."""
        ts = timestamp if timestamp is not None else time.time()
        newly: List[str] = []
        with self._lock:
            for doc_id in doc_ids:
                self._ensure_entity("doc", doc_id, ts)
                cursor = self._conn.execute(
                    "UPDATE trust_entities SET status = 'QUARANTINED', "
                    "quarantined_at = ?, updated_at = ? "
                    "WHERE entity_id = ? AND entity_type = 'doc' "
                    "AND status NOT IN ('QUARANTINED', 'REJECTED')",
                    (ts, ts, doc_id),
                )
                if cursor.rowcount and cursor.rowcount > 0:
                    newly.append(doc_id)
            self._conn.commit()
            self._blocked_cache = None
        if newly:
            logger.info("trust: quarantined %d passages: %s", len(newly), newly[:5])
        return newly

    def get_status(self, doc_id: str) -> TrustStatus:
        """Get current lifecycle status of a document."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM trust_entities "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (doc_id,),
            ).fetchone()
        if row is None:
            return TrustStatus.TRUSTED
        try:
            return TrustStatus(row["status"])
        except ValueError:
            return TrustStatus.TRUSTED

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _get_mappings(self, doc_ids: List[str]) -> Dict[str, tuple]:
        """Get (source_id, family_id) for each doc_id."""
        result: Dict[str, tuple] = {}
        # Registry first
        for start in range(0, len(doc_ids), 400):
            batch = doc_ids[start:start + 400]
            ph = ",".join("?" * len(batch))
            for row in self._conn.execute(
                f"SELECT doc_id, source_id, family_id FROM doc_registry WHERE doc_id IN ({ph})",
                batch,
            ).fetchall():
                result[row["doc_id"]] = (row["source_id"], row["family_id"])

        # Provenance store fallback for unknowns
        missing = [d for d in doc_ids if d not in result]
        if missing and self.store is not None:
            records = self.store.get_chunks(missing)
            for chunk_id, record in records.items():
                result[chunk_id] = (record.source_id, record.family_id)
                self._conn.execute(
                    "INSERT OR IGNORE INTO doc_registry(doc_id, source_id, family_id) VALUES(?,?,?)",
                    (chunk_id, record.source_id, record.family_id),
                )
        return result

    def _batch_get_entities(self, doc_ids: List[str],
                            mappings: Dict[str, tuple]) -> Dict[tuple, Dict[str, Any]]:
        """Batch-fetch trust data for all needed entities."""
        by_type: Dict[str, List[str]] = {"doc": list(doc_ids)}
        for source_id, family_id in mappings.values():
            by_type.setdefault("family", []).append(family_id)
            by_type.setdefault("source", []).append(source_id)
        # Deduplicate
        for k in by_type:
            by_type[k] = list(dict.fromkeys(by_type[k]))

        result: Dict[tuple, Dict[str, Any]] = {}
        for entity_type, ids in by_type.items():
            for start in range(0, len(ids), 400):
                batch = ids[start:start + 400]
                ph = ",".join("?" * len(batch))
                for row in self._conn.execute(
                    f"SELECT entity_id, alpha, beta, status, n_observations "
                    f"FROM trust_entities WHERE entity_type = ? AND entity_id IN ({ph})",
                    [entity_type] + batch,
                ).fetchall():
                    a, b = float(row["alpha"]), float(row["beta"])
                    result[(entity_type, row["entity_id"])] = {
                        "trust": a / (a + b) if (a + b) > 0 else 0.5,
                        "alpha": a,
                        "beta": b,
                        "status": row["status"],
                        "n_observations": int(row["n_observations"]),
                    }
        return result

    def _ensure_entity(self, entity_type: str, entity_id: str, ts: float,
                       source_id: Optional[str] = None,
                       is_burst: bool = False) -> None:
        """Create an entity record if it doesn't exist.

        For documents, the initial prior is inherited from the source:
            alpha_0 = kappa * T_source + 1
            beta_0  = kappa * (1 - T_source) + 1
        This connects new docs to source reputation (plan Section 4.6).
        """
        alpha_0 = 1.0
        beta_0 = 1.0

        if entity_type == "doc" and source_id is not None:
            # Inherit prior from source trust
            src_row = self._conn.execute(
                "SELECT alpha, beta FROM trust_entities "
                "WHERE entity_id = ? AND entity_type = 'source'",
                (source_id,),
            ).fetchone()
            if src_row:
                t_src = float(src_row["alpha"]) / (float(src_row["alpha"]) + float(src_row["beta"]))
            else:
                t_src = 0.5  # neutral prior for unknown sources
            # Apply kappa-scaled prior
            kappa = self.config.kappa
            alpha_0 = kappa * t_src + 1.0
            beta_0 = kappa * (1.0 - t_src) + 1.0
            # Burst discount: sources in ingestion bursts get weaker prior
            if is_burst:
                discount = self.config.burst_trust_discount
                alpha_0 = 1.0 + (alpha_0 - 1.0) * (1.0 - discount)
                beta_0 = beta_0 + discount * kappa * 0.5

        burst_flag = 1 if is_burst else 0
        self._conn.execute(
            "INSERT OR IGNORE INTO trust_entities"
            "(entity_id, entity_type, alpha, beta, status, n_observations, n_refutations, "
            "created_at, updated_at, is_burst_source) VALUES(?, ?, ?, ?, 'TRUSTED', 0, 0, ?, ?, ?)",
            (entity_id, entity_type, alpha_0, beta_0, ts, ts, burst_flag),
        )

    def _register_doc(self, doc_id: str, source_id: str, family_id: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO doc_registry(doc_id, source_id, family_id) VALUES(?,?,?)",
            (doc_id, source_id, family_id),
        )

    def _update_entity(self, entity_type: str, entity_id: str,
                       alpha_delta: float = 0.0, beta_delta: float = 0.0,
                       ts: float = 0.0, is_refutation: bool = False,
                       source_id: Optional[str] = None,
                       is_burst: bool = False) -> None:
        self._ensure_entity(entity_type, entity_id, ts,
                            source_id=source_id, is_burst=is_burst)
        refute_inc = 1 if is_refutation else 0
        self._conn.execute(
            "UPDATE trust_entities SET "
            "alpha = alpha + ?, beta = beta + ?, "
            "n_observations = n_observations + 1, "
            "n_refutations = n_refutations + ?, "
            "updated_at = ? "
            "WHERE entity_id = ? AND entity_type = ?",
            (alpha_delta, beta_delta, refute_inc, ts, entity_id, entity_type),
        )
        # Track global observations for decay scheduling
        self._global_obs_count += 1
        if (self.config.decay_gamma < 1.0
                and self._global_obs_count % self.config.decay_interval == 0):
            self._apply_decay(ts)

    def _check_demotion(self, doc_id: str, ts: float,
                        trigger: str = "REFUTE") -> None:
        """After a REFUTE or T_eff drop, cascade through state transitions.

        Uses a loop so that with aggressive thresholds (e.g.
        ``quarantine_refutations=1``), a single refute can cascade
        TRUSTED -> MONITORED -> QUARANTINED in one call.

        Also checks T_eff-based transitions (plan Section 4.7):
          - T_eff < monitored_t_eff (0.60) -> MONITORED
          - T_eff < quarantine_t_eff (0.20) -> QUARANTINED
        """
        for _ in range(3):  # at most 3 transitions
            row = self._conn.execute(
                "SELECT status, n_refutations, alpha, beta FROM trust_entities "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (doc_id,),
            ).fetchone()
            if row is None:
                return

            status = row["status"]
            n_ref = int(row["n_refutations"])
            a, b = float(row["alpha"]), float(row["beta"])
            t_doc = a / (a + b) if (a + b) > 0 else 0.5
            new_status = None

            # Check refutation-count thresholds (original logic)
            if status == "TRUSTED" and n_ref >= 1:
                new_status = "MONITORED"
            elif status == "MONITORED" and n_ref >= self.config.quarantine_refutations:
                new_status = "QUARANTINED"
            elif status == "QUARANTINED" and n_ref >= self.config.reject_refutations:
                new_status = "REJECTED"

            # Also check T_eff-based thresholds (plan Section 4.7)
            if new_status is None:
                if status == "TRUSTED" and t_doc < self.config.monitored_t_eff:
                    new_status = "MONITORED"
                elif status == "MONITORED" and t_doc < self.config.quarantine_t_eff:
                    new_status = "QUARANTINED"

            if new_status is None:
                return  # no more transitions

            # Write audit log entry
            self._log_transition(doc_id, "doc", status, new_status, trigger,
                                 f"n_ref={n_ref} t_doc={t_doc:.4f}", ts)

            if new_status == "QUARANTINED":
                self._conn.execute(
                    "UPDATE trust_entities SET status = 'QUARANTINED', "
                    "quarantined_at = ?, updated_at = ? "
                    "WHERE entity_id = ? AND entity_type = 'doc'",
                    (ts, ts, doc_id),
                )
            else:
                self._conn.execute(
                    "UPDATE trust_entities SET status = ?, updated_at = ? "
                    "WHERE entity_id = ? AND entity_type = 'doc'",
                    (new_status, ts, doc_id),
                )
            logger.info("trust: %s %s -> %s (refutations=%d, t_doc=%.3f)",
                        doc_id, status, new_status, n_ref, t_doc)

    def _check_recovery(self, doc_id: str, ts: float) -> None:
        """After a SUPPORT, check whether the doc can be promoted back."""
        row = self._conn.execute(
            "SELECT status, alpha, beta FROM trust_entities "
            "WHERE entity_id = ? AND entity_type = 'doc'",
            (doc_id,),
        ).fetchone()
        if row is None:
            return
        if row["status"] != "MONITORED":
            return  # QUARANTINED and REJECTED never recover automatically
        a, b = float(row["alpha"]), float(row["beta"])
        trust = a / (a + b) if (a + b) > 0 else 0.5
        if trust >= self.config.recovery_threshold:
            self._log_transition(doc_id, "doc", "MONITORED", "TRUSTED",
                                 "SUPPORT", f"trust={trust:.4f}", ts)
            self._conn.execute(
                "UPDATE trust_entities SET status = 'TRUSTED', updated_at = ? "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (ts, doc_id),
            )
            logger.info("trust: %s MONITORED -> TRUSTED (trust=%.3f)", doc_id, trust)

    # ------------------------------------------------------------------
    #  Trust decay (plan Section 4.6)
    # ------------------------------------------------------------------

    def _apply_decay(self, ts: float) -> None:
        """Apply forgetting to all entities.

        Formula from plan: alpha <- 1 + gamma * (alpha - 1)
                           beta  <- 1 + gamma * (beta  - 1)

        This pulls alpha and beta back towards 1.0 (the neutral prior),
        meaning old evidence fades and new evidence matters more.
        """
        gamma = self.config.decay_gamma
        self._conn.execute(
            "UPDATE trust_entities SET "
            "alpha = 1.0 + ? * (alpha - 1.0), "
            "beta  = 1.0 + ? * (beta  - 1.0), "
            "updated_at = ?",
            (gamma, gamma, ts),
        )
        logger.debug("trust: applied decay gamma=%.3f to all entities", gamma)

    # ------------------------------------------------------------------
    #  Structured audit log (plan B4)
    # ------------------------------------------------------------------

    def _log_transition(self, entity_id: str, entity_type: str,
                        old_status: str, new_status: str,
                        trigger: str, evidence: str, ts: float) -> None:
        """Record a state transition in the audit_log table."""
        self._conn.execute(
            "INSERT INTO audit_log(entity_id, entity_type, old_status, new_status, "
            "trigger, evidence, timestamp) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (entity_id, entity_type, old_status, new_status, trigger, evidence, ts),
        )

    def get_audit_log(self, entity_id: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Query the audit log, optionally filtered by entity."""
        with self._lock:
            if entity_id:
                rows = self._conn.execute(
                    "SELECT * FROM audit_log WHERE entity_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (entity_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    #  Burst-aware source registration (plan B2)
    # ------------------------------------------------------------------

    def mark_burst_source(self, source_id: str,
                          timestamp: Optional[float] = None) -> None:
        """Mark a source as arriving in an ingestion burst.

        Future documents from this source will get a discounted prior,
        reducing their influence on corroboration (Sybil defence).
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._ensure_entity("source", source_id, ts)
            self._conn.execute(
                "UPDATE trust_entities SET is_burst_source = 1, updated_at = ? "
                "WHERE entity_id = ? AND entity_type = 'source'",
                (ts, source_id),
            )
            self._conn.commit()
        logger.info("trust: marked source %s as burst (discounted prior)", source_id)

    def _is_burst_source(self, source_id: str) -> bool:
        """Check if a source is marked as a burst source."""
        row = self._conn.execute(
            "SELECT is_burst_source FROM trust_entities "
            "WHERE entity_id = ? AND entity_type = 'source'",
            (source_id,),
        ).fetchone()
        return bool(row and row["is_burst_source"])

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "TrustLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
