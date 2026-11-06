"""Real provenance: ingest NELA-GT news articles under their *real* outlets and publication times.

NELA-GT-2020 (Gruppi et al.; https://arxiv.org/abs/2102.04567) is a public corpus of ~1.8M news
articles from 519 real outlets, distributed as a SQLite database (table ``newsdata``) and as one
JSON file per outlet.  Per the paper each article has ``id, date, source, title, content, author,
published, published_utc, collection_utc, url``, and outlets carry reliability labels derived from
Media Bias/Fact Check.  The label file's name and columns are **not** specified in the paper, so
:func:`load_source_labels` accepts any CSV with a source column and a label column and is explicit
about the label vocabulary.  Check the dataset's licence/terms before redistribution.

Why this matters for the project: ``SourceAssigner`` invents contributors; here the contributor
is the real outlet, the timestamp is the real publication time, and the reliability label gives an
*independent*, externally produced ground truth for "is this source trustworthy" that the system
never sees.  It is stored in document metadata for evaluation only.

Nothing here downloads data: obtain the dataset yourself and pass its path.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

LABELS = ("reliable", "mixed", "unreliable")
_SOURCE_COLS = ("source", "outlet", "name", "domain")
_LABEL_COLS = ("label", "reliability", "factuality", "credibility")


def load_source_labels(path: str | Path, int_map: Optional[Mapping[int, str]] = None) -> Dict[str, str]:
    """``{outlet: "reliable" | "mixed" | "unreliable"}`` from a CSV with a header row.

    Integer labels are refused unless ``int_map`` (e.g. ``{0: "reliable", 1: "mixed", 2: "unreliable"}``)
    says what they mean: guessing a label convention would silently corrupt the ground truth.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {f.lower().strip(): f for f in (reader.fieldnames or [])}
        src = next((fields[c] for c in _SOURCE_COLS if c in fields), None)
        lab = next((fields[c] for c in _LABEL_COLS if c in fields), None)
        if src is None or lab is None:
            raise ValueError(f"{path}: need a source column {list(_SOURCE_COLS)} and a label column "
                             f"{list(_LABEL_COLS)}; found {reader.fieldnames}")
        out: Dict[str, str] = {}
        for row in reader:
            raw = str(row[lab]).strip().lower()
            if raw.lstrip("-").isdigit():
                if int_map is None:
                    raise ValueError(f"integer label {raw!r} in {path}: pass int_map to say what the "
                                     f"numbers mean")
                raw = int_map[int(raw)]
            if raw not in LABELS:
                raise ValueError(f"unknown label {raw!r} (expected one of {LABELS})")
            out[str(row[src]).strip()] = raw
    return out


def _to_epoch(row: Mapping[str, Any]) -> float:
    """Publication time as a UNIX timestamp (``published_utc``, else ``date``)."""
    for key in ("published_utc", "date", "published"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
            return number / 1000.0 if number > 1e11 else number          # tolerate milliseconds
        except (TypeError, ValueError):
            pass
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    raise ValueError(f"article {row.get('id')!r} has no usable publication time")


def iter_nela(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Stream articles from the SQLite database (``newsdata`` table) or a directory of per-outlet JSON files."""
    path = Path(path)
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            with open(file, "r", encoding="utf-8") as handle:
                articles = json.load(handle)
            for article in articles:
                article.setdefault("source", file.stem)
                yield article
        return
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT * FROM newsdata"):
            yield dict(row)
    finally:
        connection.close()


def ingest_nela(ingestor, path: str | Path, labels: Optional[Mapping[str, str]] = None,  # type: ignore[no-untyped-def]
                max_docs: Optional[int] = None, per_source_limit: Optional[int] = None,
                min_chars: int = 300, only_labelled: bool = False) -> Dict[str, Any]:
    """Ingest articles with real ``source_id`` (the outlet) and ``ingested_at`` (publication time).

    ``labels`` (from :func:`load_source_labels`) are stored as ``metadata["reliability"]`` for the
    evaluator; the pipeline, ledger and defences never read it.  Returns counts per label/outlet.
    """
    seen: Counter = Counter()
    by_label: Counter = Counter()
    n = skipped = 0
    for article in iter_nela(path):
        source = str(article.get("source") or "").strip()
        text = str(article.get("content") or "")
        label = (labels or {}).get(source)
        if (not source or len(text) < min_chars or (only_labelled and label is None)
                or (per_source_limit is not None and seen[source] >= per_source_limit)):
            skipped += 1
            continue
        try:
            ts = _to_epoch(article)
        except ValueError:
            skipped += 1
            continue
        ingestor.ingest_text(
            doc_id=f"nela_{article.get('id')}", text=text, source_id=source,
            title=str(article.get("title") or ""), origin="nela-gt", ingested_at=ts,
            metadata={"reliability": label, "url": article.get("url"), "real_provenance": True})
        seen[source] += 1
        by_label[label or "unlabelled"] += 1
        n += 1
        if max_docs is not None and n >= max_docs:
            break
    return {"documents": n, "skipped": skipped, "sources": len(seen), "by_label": dict(by_label),
            "per_source": dict(seen)}
