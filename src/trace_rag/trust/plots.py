"""Trust-dynamics plots for the gradual-poisoning experiment (plan B6).

Numbers come from the ledger's own ``trust_history`` table, so these plots work
for any run that used a :class:`~trace_rag.trust.ledger.TrustLedger` - Person C's
harness does not have to export anything extra.  The history records every
update (document, family and source, plus administrator actions), which is what
"trust-dynamics plots for the gradual-poisoning experiment" needs: trust as a
function of the query stream, with the quarantine point marked.

``matplotlib`` is imported inside the plotting functions, not at module import,
so the core package still installs and runs without it.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils.logging import get_logger

logger = get_logger(__name__)

_HISTORY_FIELDS = ["timestamp", "entity_id", "entity_type", "trust", "alpha",
                   "beta", "status", "n_observations"]


def write_history_csv(history: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    """Write exported trust history to CSV (plot-ready, no dependencies)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HISTORY_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in _HISTORY_FIELDS})
    return out


def plot_trust_dynamics(history: Sequence[Dict[str, Any]], out_dir: str | Path,
                        entity_types: Sequence[str] = ("doc", "source"),
                        entity_ids: Optional[Sequence[str]] = None,
                        dpi: int = 150) -> List[Path]:
    """One trust-over-time PNG per entity.  Returns the files written.

    The first QUARANTINED/REJECTED point in an entity's series is drawn as a
    dashed vertical line, so detection delay is visible in the figure itself.
    """
    plt = _pyplot()
    rows = [row for row in history
            if row.get("entity_type") in entity_types
            and (entity_ids is None or row.get("entity_id") in entity_ids)]
    series: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((row["entity_type"], row["entity_id"]), []).append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for (entity_type, entity_id), points in sorted(series.items()):
        points = sorted(points, key=lambda p: p["timestamp"])
        xs = [float(p["timestamp"]) for p in points]
        ys = [float(p["trust"]) for p in points]

        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2,
                label=f"{entity_type}:{entity_id}")
        bad = [p for p in points if p.get("status") in ("QUARANTINED", "REJECTED")]
        if bad:
            ax.axvline(float(bad[0]["timestamp"]), color="crimson",
                       linestyle="--", linewidth=1.0,
                       label=f"{bad[0]['status']} @ {bad[0]['timestamp']:.0f}")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("time")
        ax.set_ylabel("trust")
        ax.set_title(f"{entity_type} {entity_id}")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        path = out / f"trust_{entity_type}_{_safe(entity_id)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)
    if not written:
        logger.info("plot_trust_dynamics: nothing to plot (history empty)")
    return written


def plot_quarantine_timeline(audit_log: Sequence[Dict[str, Any]],
                             path: str | Path, dpi: int = 150) -> Optional[Path]:
    """Cumulative quarantined/rejected documents over time.

    This is the detection-delay / exposure-window figure: how many passages were
    blocked as the query stream progressed.  Returns None when the audit log
    contains no blocking transition.
    """
    events = sorted(
        (event for event in audit_log
         if event.get("new_status") in ("QUARANTINED", "REJECTED")),
        key=lambda event: float(event["timestamp"]),
    )
    if not events:
        return None
    plt = _pyplot()
    xs = [float(event["timestamp"]) for event in events]
    ys = list(range(1, len(events) + 1))

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.step(xs, ys, where="post", linewidth=1.4, color="darkred")
    ax.set_xlabel("time")
    ax.set_ylabel("blocked passages (cumulative)")
    ax.set_title("Quarantine timeline")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _pyplot():
    """Import matplotlib lazily, with a message that says how to fix it."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - only without matplotlib
        raise RuntimeError(
            "trust plots need matplotlib (pip install matplotlib)"
        ) from exc
    return plt


def _safe(name: str) -> str:
    """Filesystem-safe form of an entity id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "entity"
