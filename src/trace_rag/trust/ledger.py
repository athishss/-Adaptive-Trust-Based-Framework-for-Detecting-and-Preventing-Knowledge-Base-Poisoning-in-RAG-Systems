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
    - **Trust decay**: gamma-based forgetting per N *queries* (``begin_query``)
    - **Burst-aware prior**: sources that register several documents inside a
      short window are detected from the ledger's own registration history and
      every document of that burst starts from a discounted prior
    - **Cold-start influence**: new sources start neutral but their evidence
      counts for less until they age or accumulate verified observations
    - **Structured audit log**: every state transition is recorded with evidence
    - **Trust history**: every update is appended for the B6 dynamics plots

State machine (per document):
    TRUSTED -(refute)-> MONITORED -(2nd refute)-> QUARANTINED -(admin)-> REJECTED
       ^                     ^
       +-(support, t>0.5)----+

REJECTED is an administrator decision (plan Section 4.7).  A document that
collects ``reject_refutations`` refutations while QUARANTINED raises a review
request instead of rejecting itself; ``admin_approve_rejection`` confirms it and
``admin_approve_recovery`` is the only way back out of QUARANTINED.
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
    burst_group     INTEGER,
    PRIMARY KEY (entity_id, entity_type)
);

CREATE TABLE IF NOT EXISTS trust_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_review (
    entity_id   TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL DEFAULT 'doc',
    reason      TEXT,
    raised_at   REAL NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS trust_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    trust           REAL NOT NULL,
    alpha           REAL NOT NULL,
    beta            REAL NOT NULL,
    status          TEXT NOT NULL,
    n_observations  INTEGER NOT NULL,
    timestamp       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_entity ON trust_history(entity_id, timestamp);

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

    # ---- Bounded asymmetric Beta updates (plan Section 4.6) ----
    # A verifier's support/refute mass is the strength r in the update rule.
    # Refutation is deliberately stronger than support; every individual delta
    # is capped so one unusually confident comparison cannot dominate the ledger.
    w_s: float = 1.0
    w_r: float = 2.0
    w_max: float = 5.0
    support_family_alpha: float = 0.5
    support_source_alpha: float = 0.3
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
    decay_interval: int = 100          # apply decay every N queries (begin_query)

    # ---- State machine thresholds ----
    quarantine_refutations: int = 2    # refutations for MONITORED -> QUARANTINED
    reject_refutations: int = 3        # refutations for raising an admin review

    # ---- Hierarchical trust switch (V2 vs V3/V4 ablation, B6) ----
    # False = document-level trust only: no family/source updates, no
    # source-inherited prior and no source clamp on t_eff.
    hierarchical: bool = True
    recovery_threshold: float = 0.60   # T_eff >= this: MONITORED -> TRUSTED
    quarantine_t_eff: float = 0.20     # T_eff below this -> QUARANTINED (Section 4.7)
    monitored_t_eff: float = 0.60      # T_eff below this -> MONITORED (Section 4.7)

    # ---- HIGH-band passive penalty ----
    high_band_beta_penalty: float = 0.3

    # ---- Burst-aware Sybil defence (Section 4.5 / 4.6 / B2) ----
    burst_trust_discount: float = 0.3  # burst sources: prior reduced by this factor
    burst_window_seconds: float = 60.0  # documents from one source inside this window...
    burst_min_docs: int = 4             # ...this many of them are one ingestion burst

    # ---- Cold-start influence (Section 4.6 / B2) ----
    # New sources start neutral; what is capped is how much their evidence can
    # move corroboration mass, until they age or build verified history.
    cold_start_influence: float = 0.25      # influence multiplier for a brand-new source
    source_age_ramp_hours: float = 24.0     # age at which a source reaches full influence
    cold_start_min_observations: int = 5    # verified observations that also mature a source

    # ---- Trust history for the B6 dynamics plots ----
    record_history: bool = True

    def __post_init__(self) -> None:
        """Reject invalid trust settings before they can corrupt the ledger."""
        import math

        positive = {
            "w_s": self.w_s,
            "w_r": self.w_r,
            "w_max": self.w_max,
            "prior_weight": self.prior_weight,
            "decay_interval": self.decay_interval,
            "burst_window_seconds": self.burst_window_seconds,
            "burst_min_docs": self.burst_min_docs,
            "source_age_ramp_hours": self.source_age_ramp_hours,
            "cold_start_min_observations": self.cold_start_min_observations,
            "quarantine_refutations": self.quarantine_refutations,
            "reject_refutations": self.reject_refutations,
        }
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in positive.values()):
            bad = [name for name, value in positive.items()
                   if not math.isfinite(float(value)) or float(value) <= 0]
            raise ValueError(f"trust settings must be positive finite values: {bad}")
        if self.w_r <= self.w_s:
            raise ValueError("w_r must be greater than w_s")
        if not 0.0 <= self.decay_gamma <= 1.0:
            raise ValueError("decay_gamma must be in [0, 1]")
        if self.kappa < 0 or not math.isfinite(self.kappa):
            raise ValueError("kappa must be finite and non-negative")
        if not 0.0 <= self.t_cap <= 1.0:
            raise ValueError("t_cap must be in [0, 1]")
        if not 0.0 <= self.cold_start_influence <= 1.0:
            raise ValueError("cold_start_influence must be in [0, 1]")
        for name in ("support_family_alpha", "support_source_alpha",
                     "refute_family_beta", "refute_source_beta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not (0.0 <= self.quarantine_t_eff < self.monitored_t_eff <= 1.0):
            raise ValueError("trust-state thresholds must satisfy 0 <= quarantine < monitored <= 1")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery_threshold must be in [0, 1]")
        if not 0.0 <= self.burst_trust_discount <= 1.0:
            raise ValueError("burst_trust_discount must be in [0, 1]")


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
        self._query_count: int = 0        # persisted decay counter (per query, plan 4.6)

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
            for col, defn in [("is_burst_source", "INTEGER NOT NULL DEFAULT 0"),
                              ("burst_group", "INTEGER")]:
                try:
                    self._conn.execute(
                        f"ALTER TABLE trust_entities ADD COLUMN {col} {defn}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            self._conn.execute(
                "INSERT OR IGNORE INTO trust_meta(key, value) VALUES('query_count', '0')"
            )
            row = self._conn.execute(
                "SELECT value FROM trust_meta WHERE key = 'query_count'"
            ).fetchone()
            self._query_count = int(row["value"]) if row is not None else 0
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
            # Materialise priors lazily when a passage first enters retrieval.
            # This is initialization, not evidence: only SUPPORT/REFUTE outcomes
            # may change alpha/beta. It ensures a new document immediately
            # inherits its source prior and starts in the policy-appropriate band.
            records = self.store.get_chunks(doc_ids) if self.store is not None else {}
            source_times: Dict[str, float] = {}
            if self.store is not None:
                for source_id in {value[0] for value in mappings.values()}:
                    stats = self.store.source_stats(source_id)
                    if stats is not None:
                        source_times[source_id] = float(stats.first_seen)
            for doc_id in doc_ids:
                source_id, family_id = mappings.get(doc_id, ("unknown", "unknown"))
                if source_id == "unknown" or family_id == "unknown":
                    continue
                record = records.get(doc_id)
                doc_time = float(record.ingested_at) if record is not None else now
                source_time = source_times.get(source_id, now)
                if self.config.hierarchical:
                    self._ensure_entity("source", source_id, source_time)
                    self._ensure_entity("family", family_id, doc_time)
                self._register_doc(doc_id, source_id, family_id)
                is_burst = self._detect_burst(source_id, doc_time)
                self._ensure_entity("doc", doc_id, doc_time, source_id=source_id,
                                    is_burst=is_burst)
            # A shared family/source can change a document's effective trust
            # between its own observations. Persist that lifecycle transition
            # here so blocked_doc_ids(), audit history and admin recovery all
            # agree with the snapshot used by retrieval.
            for doc_id in doc_ids:
                if doc_id in mappings:
                    self._check_demotion(doc_id, now, trigger="T_EFF")
            self._conn.commit()
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
            if self.config.hierarchical:
                # Plan Section 4.6: T_eff = w * T_doc + (1 - w) * min(T_source, T_family).
                # The min matters: averaging the two let a poisoned family hide
                # behind a clean source, and a poisoned source behind a clean
                # family, which is exactly the inheritance this is defending.
                t_parents = min(t_family, t_source)
            else:
                t_parents = 0.5  # ablation: document-level trust only
            t_eff = w_doc * t_doc + (1.0 - w_doc) * t_parents
            if self.config.hierarchical:
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
                           outcome: str, timestamp: Optional[float] = None,
                           strength: float = 1.0) -> None:
        """Apply a corroborated verifier outcome to the trust hierarchy.

        NEUTRAL is deliberately a no-op: it does not create reputation, change
        alpha/beta, increment observation counts, or alter lifecycle state.
        The verifier's trust-weighted evidence mass is ``strength``; support and
        refutation deltas are asymmetric and individually bounded by ``w_max``.
        """
        if outcome not in ("SUPPORT", "REFUTE", "NEUTRAL"):
            raise ValueError(f"outcome must be SUPPORT/REFUTE/NEUTRAL, got {outcome!r}")
        ts = timestamp if timestamp is not None else time.time()
        if outcome == "NEUTRAL":
            # Register provenance and materialise cold-start priors, but do not
            # count a verified observation or alter any existing Beta values.
            with self._lock:
                self._register_doc(doc_id, source_id, family_id)
                is_burst = self._detect_burst(source_id, ts)
                if self.config.hierarchical:
                    self._ensure_entity("source", source_id, ts)
                    self._ensure_entity("family", family_id, ts)
                self._ensure_entity("doc", doc_id, ts, source_id=source_id,
                                    is_burst=is_burst)
                self._conn.commit()
            return
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        strength = float(strength)

        with self._lock:
            self._register_doc(doc_id, source_id, family_id)
            # Burst detection runs before this document's entity is created, so
            # its prior already carries the discount, and it back-applies the
            # discount to the documents that opened the burst (plan B2/S4).
            is_burst = self._detect_burst(source_id, ts)
            support_base = min(self.config.w_s * strength, self.config.w_max)
            refute_base = min(self.config.w_r * strength, self.config.w_max)

            if outcome == "SUPPORT":
                self._update_entity("doc", doc_id,
                                    alpha_delta=support_base, ts=ts,
                                    source_id=source_id, is_burst=is_burst)
                if self.config.hierarchical:
                    self._update_entity("family", family_id,
                                        alpha_delta=min(support_base * self.config.support_family_alpha,
                                                        self.config.w_max), ts=ts)
                    self._update_entity("source", source_id,
                                        alpha_delta=min(support_base * self.config.support_source_alpha,
                                                        self.config.w_max), ts=ts)
            else:  # REFUTE
                self._update_entity("doc", doc_id,
                                    beta_delta=refute_base, ts=ts,
                                    is_refutation=True,
                                    source_id=source_id, is_burst=is_burst)
                if self.config.hierarchical:
                    self._update_entity("family", family_id,
                                        beta_delta=min(refute_base * self.config.refute_family_beta,
                                                       self.config.w_max), ts=ts,
                                        is_refutation=True)
                    self._update_entity("source", source_id,
                                        beta_delta=min(refute_base * self.config.refute_source_beta,
                                                       self.config.w_max), ts=ts,
                                        is_refutation=True)

            # State machine transitions are based only on a verified outcome.
            if outcome == "REFUTE":
                self._check_demotion(doc_id, ts)
            else:
                self._check_recovery(doc_id, ts)

            self._conn.commit()
            self._blocked_cache = None  # invalidate

    def record_high_band(self, doc_id: str, source_id: str, family_id: str,
                         timestamp: Optional[float] = None) -> None:
        """Deprecated no-op: suspicion alone must never update trust.

        HIGH-band passages are excluded from the current answer and queued for
        off-path verification. Only that verifier's SUPPORT/REFUTE outcome may
        write to the trust ledger.
        """
        del doc_id, source_id, family_id, timestamp
        logger.warning("ignoring unverified HIGH-band trust update; enqueue it for verification")

    def quarantine(self, doc_ids: Sequence[str], reason: str = "quarantined",
                   timestamp: Optional[float] = None) -> List[str]:
        """Directly quarantine specific doc_ids.  Returns the list actually transitioned."""
        ts = timestamp if timestamp is not None else time.time()
        newly: List[str] = []
        with self._lock:
            for doc_id in doc_ids:
                self._ensure_entity("doc", doc_id, ts)
                row = self._conn.execute(
                    "SELECT status FROM trust_entities "
                    "WHERE entity_id = ? AND entity_type = 'doc'", (doc_id,),
                ).fetchone()
                old_status = row["status"] if row is not None else "TRUSTED"
                cursor = self._conn.execute(
                    "UPDATE trust_entities SET status = 'QUARANTINED', "
                    "quarantined_at = ?, updated_at = ? "
                    "WHERE entity_id = ? AND entity_type = 'doc' "
                    "AND status NOT IN ('QUARANTINED', 'REJECTED')",
                    (ts, ts, doc_id),
                )
                if cursor.rowcount and cursor.rowcount > 0:
                    self._log_transition(doc_id, "doc", old_status, "QUARANTINED",
                                         "ADMIN_OR_POLICY", str(reason), ts)
                    newly.append(doc_id)
            self._conn.commit()
            self._blocked_cache = None
        if newly:
            logger.info("trust: quarantined %d passages: %s", len(newly), newly[:5])
        return newly

    def get_status(self, doc_id: str) -> TrustStatus:
        """Get the current status, including an effective-trust cold-start state."""
        snapshot = self.get_trust([doc_id]).get(doc_id)
        return snapshot.status if snapshot is not None else TrustStatus.MONITORED

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

        if entity_type == "doc" and source_id is not None and self.config.hierarchical:
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
        if self.config.record_history:
            self._record_history(entity_id, entity_type, ts)

    def _record_history(self, entity_id: str, entity_type: str, ts: float) -> None:
        """Append one point to the trust history (plan B6 trust-dynamics plots)."""
        row = self._conn.execute(
            "SELECT alpha, beta, status, n_observations FROM trust_entities "
            "WHERE entity_id = ? AND entity_type = ?",
            (entity_id, entity_type),
        ).fetchone()
        if row is None:
            return
        a, b = float(row["alpha"]), float(row["beta"])
        self._conn.execute(
            "INSERT INTO trust_history(entity_id, entity_type, trust, alpha, beta, "
            "status, n_observations, timestamp) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (entity_id, entity_type, a / (a + b) if (a + b) > 0 else 0.5,
             a, b, row["status"], int(row["n_observations"]), ts),
        )

    def _effective_trust_for_doc(self, doc_id: str) -> float:
        """Compute hierarchical T_eff from the current persisted Beta values."""
        doc = self._conn.execute(
            "SELECT alpha, beta, n_observations FROM trust_entities "
            "WHERE entity_id = ? AND entity_type = 'doc'", (doc_id,),
        ).fetchone()
        if doc is None:
            return 0.5
        alpha, beta = float(doc["alpha"]), float(doc["beta"])
        t_doc = alpha / (alpha + beta) if alpha + beta > 0 else 0.5
        n_obs = int(doc["n_observations"])
        t_source = t_family = 0.5
        if self.config.hierarchical:
            mapping = self._conn.execute(
                "SELECT source_id, family_id FROM doc_registry WHERE doc_id = ?", (doc_id,),
            ).fetchone()
            if mapping is not None:
                for entity_type, entity_id in (("source", mapping["source_id"]),
                                               ("family", mapping["family_id"])):
                    parent = self._conn.execute(
                        "SELECT alpha, beta FROM trust_entities "
                        "WHERE entity_type = ? AND entity_id = ?",
                        (entity_type, entity_id),
                    ).fetchone()
                    trust = (float(parent["alpha"]) / (float(parent["alpha"]) + float(parent["beta"]))
                             if parent is not None else 0.5)
                    if entity_type == "source":
                        t_source = trust
                    else:
                        t_family = trust
        parent_trust = min(t_source, t_family) if self.config.hierarchical else 0.5
        weight = min(1.0, n_obs / max(1.0, self.config.prior_weight))
        t_eff = weight * t_doc + (1.0 - weight) * parent_trust
        if self.config.hierarchical:
            t_eff = min(t_eff, t_source)
        if n_obs < self.config.prior_weight:
            t_eff = min(t_eff, self.config.t_cap)
        return float(max(0.0, min(1.0, t_eff)))

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
            t_eff = self._effective_trust_for_doc(doc_id)
            new_status = None

            if status == "TRUSTED" and (
                    (trigger == "REFUTE" and n_ref >= 1)
                    or t_eff < self.config.monitored_t_eff):
                new_status = "MONITORED"
            elif status == "MONITORED" and (
                    (trigger == "REFUTE" and n_ref >= self.config.quarantine_refutations)
                    or t_eff < self.config.quarantine_t_eff):
                new_status = "QUARANTINED"
            elif status == "QUARANTINED" and n_ref >= self.config.reject_refutations:
                # REJECTED is an administrator decision; keep the document
                # quarantined until a human approves the rejection.
                self._raise_review(doc_id, f"n_ref={n_ref} t_eff={t_eff:.4f}", ts)
                return

            if new_status is None:
                return  # no more transitions

            # Every lifecycle transition carries both document and effective trust.
            self._log_transition(doc_id, "doc", status, new_status, trigger,
                                 f"n_ref={n_ref} t_doc={t_doc:.4f} t_eff={t_eff:.4f}", ts)

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
            self._blocked_cache = None
            logger.info("trust: %s %s -> %s (refutations=%d, t_eff=%.3f)",
                        doc_id, status, new_status, n_ref, t_eff)

    def begin_query(self, timestamp: Optional[float] = None) -> None:
        """Mark the start of one query.

        Trust decay is scheduled on queries, not on individual observations
        (plan Section 4.6, "forgetting per N queries"), so the caller -- the
        policy in the pipeline -- calls this once per query.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._query_count += 1
            if (self.config.decay_gamma < 1.0
                    and self._query_count % self.config.decay_interval == 0):
                self._apply_decay(ts)
            self._conn.execute(
                "UPDATE trust_meta SET value = ? WHERE key = 'query_count'",
                (str(self._query_count),),
            )
            self._conn.commit()

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
        trust = self._effective_trust_for_doc(doc_id)
        if trust >= self.config.recovery_threshold:
            self._log_transition(doc_id, "doc", "MONITORED", "TRUSTED",
                                 "SUPPORT", f"t_eff={trust:.4f}", ts)
            self._conn.execute(
                "UPDATE trust_entities SET status = 'TRUSTED', updated_at = ? "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (ts, doc_id),
            )
            logger.info("trust: %s MONITORED -> TRUSTED (trust=%.3f)", doc_id, trust)

    # ------------------------------------------------------------------
    #  Administrator review (plan Section 4.7)
    # ------------------------------------------------------------------

    def _raise_review(self, doc_id: str, reason: str, ts: float) -> None:
        """Queue a document for an administrator decision (idempotent)."""
        cursor = self._conn.execute(
            "INSERT INTO admin_review(entity_id, entity_type, reason, raised_at, state) "
            "VALUES(?, 'doc', ?, ?, 'pending') "
            "ON CONFLICT(entity_id) DO UPDATE SET "
            "reason = excluded.reason, raised_at = excluded.raised_at, state = 'pending' "
            "WHERE admin_review.state != 'pending'",
            (doc_id, reason, ts),
        )
        if cursor.rowcount:
            logger.info("trust: raised admin review for %s (%s)", doc_id, reason)

    def pending_reviews(self) -> List[Dict[str, Any]]:
        """Documents waiting for an administrator decision."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM admin_review WHERE state = 'pending' ORDER BY raised_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def admin_approve_rejection(self, doc_id: str, approved_by: str,
                                note: str = "",
                                timestamp: Optional[float] = None) -> bool:
        """Administrator confirms REJECTED for a QUARANTINED document.

        Plan Section 4.7: REJECTED means permanently removed from the index and
        the source flagged.  Returns True when the transition happened.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM trust_entities "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (doc_id,),
            ).fetchone()
            if row is None or row["status"] != "QUARANTINED":
                logger.warning("trust: cannot reject %s (status=%s)",
                               doc_id, row["status"] if row else "unknown")
                return False
            self._conn.execute(
                "UPDATE trust_entities SET status = 'REJECTED', updated_at = ? "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (ts, doc_id),
            )
            self._log_transition(doc_id, "doc", "QUARANTINED", "REJECTED",
                                 "ADMIN_APPROVAL",
                                 f"approved_by={approved_by}"
                                 + (f"; {note}" if note else ""), ts)
            # Flag the source: one extra refutation-weight penalty, so every new
            # document from a source that produced a rejected passage starts lower.
            if self.config.hierarchical:
                mapping = self._conn.execute(
                    "SELECT source_id FROM doc_registry WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if mapping is not None:
                    self._update_entity("source", mapping["source_id"],
                                        beta_delta=self.config.refute_source_beta, ts=ts,
                                        is_refutation=True)
            self._conn.execute(
                "UPDATE admin_review SET state = 'approved' WHERE entity_id = ?",
                (doc_id,),
            )
            self._conn.commit()
            self._blocked_cache = None
        logger.info("trust: %s QUARANTINED -> REJECTED (approved_by=%s)", doc_id, approved_by)
        return True

    def admin_decline_rejection(self, doc_id: str, reviewed_by: str,
                               note: str = "",
                               timestamp: Optional[float] = None) -> bool:
        """Administrator declines the rejection; the document stays QUARANTINED."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE admin_review SET state = 'declined' "
                "WHERE entity_id = ? AND state = 'pending'",
                (doc_id,),
            )
            if not cursor.rowcount:
                return False
            logger.info("trust: admin declined rejection of %s (by %s%s)",
                        doc_id, reviewed_by, f": {note}" if note else "")
        return True

    def admin_approve_recovery(self, doc_id: str, approved_by: str,
                               note: str = "",
                               timestamp: Optional[float] = None) -> bool:
        """Administrator moves a QUARANTINED document back to MONITORED.

        Plan Section 4.7: recovery needs administrator approval so an attacker
        cannot talk a document back into use.  Returns True if it transitioned.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM trust_entities "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (doc_id,),
            ).fetchone()
            if row is None or row["status"] != "QUARANTINED":
                return False
            self._conn.execute(
                "UPDATE trust_entities SET status = 'MONITORED', "
                "quarantined_at = NULL, updated_at = ? "
                "WHERE entity_id = ? AND entity_type = 'doc'",
                (ts, doc_id),
            )
            self._log_transition(doc_id, "doc", "QUARANTINED", "MONITORED",
                                 "ADMIN_RECOVERY",
                                 f"approved_by={approved_by}"
                                 + (f"; {note}" if note else ""), ts)
            self._conn.execute(
                "UPDATE admin_review SET state = 'recovered' WHERE entity_id = ?",
                (doc_id,),
            )
            self._conn.commit()
            self._blocked_cache = None
        logger.info("trust: %s QUARANTINED -> MONITORED (approved_by=%s)", doc_id, approved_by)
        return True

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

    def export_history(self, entity_id: Optional[str] = None,
                       entity_type: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export the trust history for the B6 trust-dynamics plots."""
        sql = "SELECT * FROM trust_history"
        clauses: List[str] = []
        params: List[Any] = []
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    #  Burst-aware source registration (plan B2)
    # ------------------------------------------------------------------

    def mark_burst_source(self, source_id: str,
                          timestamp: Optional[float] = None) -> None:
        """Mark a source as arriving in an ingestion burst.

        Documents from this source get a discounted prior, reducing their
        influence on corroboration (Sybil defence).  Bursts are normally
        detected automatically from the registration history (``_detect_burst``);
        this is the explicit override for a signal computed outside the ledger.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._ensure_entity("source", source_id, ts)
            group = (int(ts // self.config.burst_window_seconds)
                     if self.config.burst_window_seconds > 0 else None)
            self._conn.execute(
                "UPDATE trust_entities SET is_burst_source = 1, "
                "burst_group = COALESCE(burst_group, ?), updated_at = ? "
                "WHERE entity_id = ? AND entity_type = 'source'",
                (group, ts, source_id),
            )
            self._conn.commit()
        logger.info("trust: marked source %s as burst (discounted prior)", source_id)

    def _detect_burst(self, source_id: str, ts: float) -> bool:
        """Detect an ingestion burst from *source_id* and flag the source.

        Plan signal S4 / Section 4.6: a source that registers
        ``burst_min_docs`` documents inside ``burst_window_seconds`` is an
        ingestion burst (normal contributors do not post in tight batches).
        Detection uses the ledger's own registration history, so the defence
        fires in a real run without waiting for the ingestion layer to call
        ``mark_burst_source``.

        The flag is set when the burst becomes visible and is applied
        retroactively to the documents already registered in the window,
        because the first documents of a burst arrive before it is detectable.

        Returns True when the source is (now) flagged as a burst.
        """
        if self.config.burst_window_seconds <= 0 or self.config.burst_min_docs <= 1:
            return self._is_burst_source(source_id)
        if self._is_burst_source(source_id):
            return True

        since = ts - self.config.burst_window_seconds
        recent = int(self._conn.execute(
            "SELECT COUNT(*) AS n FROM doc_registry r "
            "JOIN trust_entities e ON e.entity_id = r.doc_id AND e.entity_type = 'doc' "
            "WHERE r.source_id = ? AND e.created_at >= ? AND e.created_at <= ?",
            (source_id, since, ts),
        ).fetchone()["n"])
        if recent + 1 < self.config.burst_min_docs:
            return False

        self._ensure_entity("source", source_id, ts)
        self._conn.execute(
            "UPDATE trust_entities SET is_burst_source = 1, "
            "burst_group = COALESCE(burst_group, ?), updated_at = ? "
            "WHERE entity_id = ? AND entity_type = 'source'",
            (int(ts // self.config.burst_window_seconds), ts, source_id),
        )
        discount = float(self.config.burst_trust_discount)
        if discount > 0.0:
            docs = self._conn.execute(
                "SELECT r.doc_id FROM doc_registry r "
                "JOIN trust_entities e ON e.entity_id = r.doc_id AND e.entity_type = 'doc' "
                "WHERE r.source_id = ? AND e.created_at >= ? AND e.created_at <= ?",
                (source_id, since, ts),
            ).fetchall()
            for doc_row in docs:
                self._conn.execute(
                    "UPDATE trust_entities SET "
                    "alpha = 1.0 + (alpha - 1.0) * ?, beta = beta + ? "
                    "WHERE entity_id = ? AND entity_type = 'doc'",
                    (1.0 - discount, discount * self.config.kappa * 0.5,
                     doc_row["doc_id"]),
                )
        logger.info("trust: detected ingestion burst from %s (%d docs in %.0fs)",
                    source_id, recent + 1, self.config.burst_window_seconds)
        return True

    def influence_factor(self, source_id: str,
                         now: Optional[float] = None) -> float:
        """How much weight this source's evidence may carry right now.

        Plan Section 4.6, B2: new sources start neutral, but their *influence*
        on corroboration mass is capped - that is the Sybil defence.  The
        factor ramps from ``cold_start_influence`` to 1.0 as the source either
        ages past ``source_age_ramp_hours`` or accumulates
        ``cold_start_min_observations`` verified observations.  A source the
        ledger has never seen has no history at all and gets the floor.
        """
        if not self.config.hierarchical:
            return 1.0
        ts = now if now is not None else time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT created_at, n_observations FROM trust_entities "
                "WHERE entity_id = ? AND entity_type = 'source'",
                (source_id,),
            ).fetchone()
            if row is None:
                # The source may be known only through its documents (NEUTRAL
                # observations never create the source entity), so age it from
                # its first registered document instead of calling it unknown.
                row = self._conn.execute(
                    "SELECT MIN(e.created_at) AS created_at, 0 AS n_observations "
                    "FROM doc_registry r JOIN trust_entities e "
                    "ON e.entity_id = r.doc_id AND e.entity_type = 'doc' "
                    "WHERE r.source_id = ?",
                    (source_id,),
                ).fetchone()
        if row is None or row["created_at"] is None:
            return max(0.0, min(1.0, float(self.config.cold_start_influence)))
        if int(row["n_observations"]) >= self.config.cold_start_min_observations:
            return 1.0
        ramp = float(self.config.source_age_ramp_hours)
        if ramp <= 0.0:
            return 1.0
        age_hours = max(0.0, (ts - float(row["created_at"])) / 3600.0)
        maturity = min(1.0, age_hours / ramp)
        floor = float(self.config.cold_start_influence)
        return max(0.0, min(1.0, floor + (1.0 - floor) * maturity))

    def same_burst(self, source_a: str, source_b: str) -> bool:
        """Whether two sources arrived in the same detected ingestion burst.

        Passed to the verifier so that passages from one burst cannot
        corroborate each other (plan Section 4.5, step 3).  Two sources share a
        burst only when both are flagged and their burst groups are equal, so
        bulk-ingested *clean* sources are not excluded from each other merely
        for arriving in batches.
        """
        if source_a == source_b:
            return True
        with self._lock:
            rows = self._conn.execute(
                "SELECT entity_id, burst_group, is_burst_source FROM trust_entities "
                "WHERE entity_type = 'source' AND entity_id IN (?, ?)",
                (source_a, source_b),
            ).fetchall()
        groups = {row["entity_id"]: row["burst_group"]
                  for row in rows if row["is_burst_source"]}
        if len(groups) < 2:
            return False
        group_a, group_b = groups.get(source_a), groups.get(source_b)
        return group_a is not None and group_a == group_b

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
