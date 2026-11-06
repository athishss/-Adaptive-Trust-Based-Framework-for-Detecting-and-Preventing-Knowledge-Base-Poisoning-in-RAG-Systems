"""Statistical analysis of a finished experiment matrix (reads ``cases/*/stream.jsonl``).

Unit of analysis
    One *(seed, target question)* pair.  Repeated asks of the same target are not
    independent samples; they are collapsed into one unit before any interval or test.
    A unit's outcome is reported three ways:

    ``first``  the first attempt after the poison was ingested (cold start: what a static
               filter can do);
    ``last``   the final attempt (after the defence has had every chance to learn);
    ``mean``   the share of attempts (what the matrix CSV already reports).

Per (attack, defence) we report ASR, accuracy under attack (gold answer delivered),
abstention and exposure, each with a percentile bootstrap CI over units.  Between
defences we run the exact McNemar test on paired binary unit outcomes, and apply the
Holm correction to every family of comparisons.  Nothing is inferred when fewer than
``min_units`` paired units exist; the row is marked ``n/a``.

Abstention counts as attack failure (it is not an attacker answer), which is why accuracy
and abstention are always shown next to ASR: a system that refuses everything has ASR 0.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .statistics import bootstrap_ci, mcnemar_exact

_CASE = re.compile(r"^(?P<attack>.+?)__(?P<baseline>.+?)__seed(?P<seed>-?\d+)$")
METRICS = ("asr", "accuracy", "abstention", "exposure")


@dataclass(frozen=True)
class Unit:
    attack: str
    baseline: str
    seed: int
    qid: str
    n_attempts: int
    asr_first: float
    asr_last: float
    asr_mean: float
    accuracy_mean: Optional[float]
    abstention_mean: float
    exposure_mean: float


def _qid(query_id: str) -> str:
    """``q12_s34`` -> ``q12`` (the harness suffixes the step)."""
    return re.sub(r"_s\d+$", "", query_id)


def load_units(results_dir: str | Path) -> List[Unit]:
    root = Path(results_dir) / "cases"
    units: List[Unit] = []
    for case_dir in sorted(root.glob("*")):
        match = _CASE.match(case_dir.name)
        stream = case_dir / "stream.jsonl"
        if not match or not stream.exists():
            continue
        by_q: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for line in stream.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Eligible attempt: a target query with poison already ingested, and a
            # measured answer.  Un-measured rows are never turned into zeros.
            if row.get("is_target") and row.get("poison_targeted") and \
                    row.get("answer_attack_status") == "MEASURED":
                by_q[_qid(row["query_id"])].append(row)
        for qid, rows in sorted(by_q.items()):
            rows.sort(key=lambda r: r["step"])
            asr = [1.0 if r["answer_attack_success"] else 0.0 for r in rows]
            acc = [1.0 if r.get("answer_gold_correct") else 0.0 for r in rows
                   if r.get("answer_gold_correct") is not None]
            units.append(Unit(
                attack=match["attack"], baseline=match["baseline"], seed=int(match["seed"]),
                qid=qid, n_attempts=len(rows), asr_first=asr[0], asr_last=asr[-1],
                asr_mean=float(np.mean(asr)),
                accuracy_mean=float(np.mean(acc)) if acc else None,
                abstention_mean=float(np.mean([1.0 if r["abstained"] else 0.0 for r in rows])),
                exposure_mean=float(np.mean([1.0 if r["poison_in_context"] else 0.0 for r in rows])),
            ))
    return units


def _ci(values: Sequence[float], seed: int) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < 2:
        return None, None
    return bootstrap_ci(values, seed=seed)


def summarise(units: Iterable[Unit], bootstrap_seed: int = 20260921) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Unit]] = defaultdict(list)
    for u in units:
        groups[(u.attack, u.baseline)].append(u)
    rows: List[Dict[str, Any]] = []
    for (attack, baseline), us in sorted(groups.items()):
        row: Dict[str, Any] = {"attack": attack, "baseline": baseline, "units": len(us),
                               "seeds": len({u.seed for u in us})}
        series = {
            "asr_mean": [u.asr_mean for u in us], "asr_first": [u.asr_first for u in us],
            "asr_last": [u.asr_last for u in us],
            "accuracy": [u.accuracy_mean for u in us if u.accuracy_mean is not None],
            "abstention": [u.abstention_mean for u in us],
            "exposure": [u.exposure_mean for u in us],
        }
        for name, values in series.items():
            lo, hi = _ci(values, bootstrap_seed)
            row[name] = float(np.mean(values)) if values else None
            row[f"{name}_ci"] = [lo, hi]
        rows.append(row)
    return rows


def holm(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni adjusted p-values; ``None`` entries are skipped, not counted."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    running, m = 0.0, len(order)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvalues[i])))
        adjusted[i] = running
    return adjusted


def paired_tests(units: Iterable[Unit], reference: str = "RAGLedger",
                 outcome: str = "asr_last", min_units: int = 10) -> List[Dict[str, Any]]:
    """McNemar (exact) between ``reference`` and every other defence, per attack."""
    if outcome not in ("asr_first", "asr_last"):
        raise ValueError("outcome must be asr_first or asr_last (binary per unit)")
    table: Dict[Tuple[str, str], Dict[Tuple[int, str], float]] = defaultdict(dict)
    for u in units:
        table[(u.attack, u.baseline)][(u.seed, u.qid)] = getattr(u, outcome)
    attacks = sorted({a for a, _ in table})
    baselines = sorted({b for _, b in table})
    rows: List[Dict[str, Any]] = []
    for attack in attacks:
        ref = table.get((attack, reference))
        if not ref:
            continue
        for baseline in baselines:
            other = table.get((attack, baseline))
            if baseline == reference or not other:
                continue
            shared = sorted(set(ref) & set(other))
            ref_only = sum(1 for k in shared if ref[k] == 1.0 and other[k] == 0.0)
            other_only = sum(1 for k in shared if ref[k] == 0.0 and other[k] == 1.0)
            p = mcnemar_exact(ref_only, other_only) if len(shared) >= min_units else None
            rows.append({"attack": attack, "reference": reference, "baseline": baseline,
                         "outcome": outcome, "paired_units": len(shared),
                         "reference_succeeds_only": ref_only, "baseline_succeeds_only": other_only,
                         "p_exact": p,
                         "reference_asr": float(np.mean([ref[k] for k in shared])) if shared else None,
                         "baseline_asr": float(np.mean([other[k] for k in shared])) if shared else None})
    adjusted = holm([r["p_exact"] for r in rows])
    for r, a in zip(rows, adjusted):
        r["p_holm"] = a
    return rows


def case_cost(results_dir: str | Path) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
    """Mean LLM calls per query and p99 latency per (attack, defence), from ``metrics.json``."""
    out: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for metrics in sorted((Path(results_dir) / "cases").glob("*/metrics.json")):
        match = _CASE.match(metrics.parent.name)
        if not match:
            continue
        m = json.loads(metrics.read_text(encoding="utf-8"))
        key = (match["attack"], match["baseline"])
        calls = (m.get("llm_calls") or {}).get("mean_per_query")
        p99 = (m.get("latency_ms") or {}).get("p99_ms")
        util = m.get("clean_utility")
        w = (m.get("withholding") or {}).get("withheld_clean_rate")
        for name, v in (("llm_calls_per_query", calls), ("p99_ms", p99), ("clean_utility", util),
                        ("withheld_clean_rate", w)):
            if v is not None:
                out[key][name].append(float(v))
    return {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in out.items()}


def _fmt(value: Optional[float], ci: Optional[Sequence[Optional[float]]] = None) -> str:
    if value is None:
        return "n/a"
    if ci and ci[0] is not None:
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.3f}"


def to_markdown(rows: Sequence[Mapping[str, Any]], tests: Sequence[Mapping[str, Any]],
                cost: Mapping[Tuple[str, str], Mapping[str, float]], scope: str) -> str:
    lines = [f"# Analysis: {scope}", "",
             "Unit = (seed, target question). CIs are 95% percentile bootstrap over units. "
             "ASR counts abstention as failure; read it with accuracy and abstention.", "",
             "| attack | defence | units | ASR first | ASR last | accuracy | abstention | exposure "
             "| LLM calls/q | clean utility |", "|---|---|---:|---|---|---|---|---|---:|---:|"]
    for r in rows:
        c = cost.get((r["attack"], r["baseline"]), {})
        lines.append("| {a} | {b} | {n} | {f} | {l} | {acc} | {ab} | {ex} | {calls} | {util} |".format(
            a=r["attack"], b=r["baseline"], n=r["units"], f=_fmt(r["asr_first"], r["asr_first_ci"]),
            l=_fmt(r["asr_last"], r["asr_last_ci"]), acc=_fmt(r["accuracy"], r["accuracy_ci"]),
            ab=_fmt(r["abstention"], r["abstention_ci"]), ex=_fmt(r["exposure"], r["exposure_ci"]),
            calls=_fmt(c.get("llm_calls_per_query")), util=_fmt(c.get("clean_utility"))))
    if tests:
        lines += ["", "## Paired exact McNemar tests (Holm-adjusted within each outcome)", "",
                  "| attack | outcome | reference | vs | paired | ref-only | other-only | p | p (Holm) |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|"]
        for t in tests:
            p = "n/a" if t["p_exact"] is None else f"{t['p_exact']:.4f}"
            ph = "n/a" if t["p_holm"] is None else f"{t['p_holm']:.4f}"
            lines.append(f"| {t['attack']} | {t['outcome']} | {t['reference']} | {t['baseline']} | "
                         f"{t['paired_units']} | {t['reference_succeeds_only']} | "
                         f"{t['baseline_succeeds_only']} | {p} | {ph} |")
    return "\n".join(lines) + "\n"


def analyse_directory(results_dir: str | Path, reference: str = "RAGLedger",
                      scope: str = "experiment", min_units: int = 10) -> Dict[str, Any]:
    units = load_units(results_dir)
    if not units:
        raise ValueError(f"no measured target units under {results_dir}/cases "
                         "(did the run supply gold answers and finish?)")
    rows = summarise(units)
    tests = (paired_tests(units, reference, "asr_last", min_units)
             + paired_tests(units, reference, "asr_first", min_units))
    # Holm within each outcome family, not across the two.
    for outcome in ("asr_last", "asr_first"):
        fam = [t for t in tests if t["outcome"] == outcome]
        for t, a in zip(fam, holm([t["p_exact"] for t in fam])):
            t["p_holm"] = a
    cost = case_cost(results_dir)
    out = Path(results_dir)
    payload = {"scope": scope, "reference": reference, "units": len(units), "summary": rows,
               "paired_tests": tests,
               "cost": {f"{a}|{b}": v for (a, b), v in cost.items()}}
    (out / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "analysis.md").write_text(to_markdown(rows, tests, cost, scope), encoding="utf-8")
    with (out / "analysis.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["attack", "baseline", "units", "seeds", "asr_first", "asr_last", "asr_mean",
                  "accuracy", "abstention", "exposure"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return payload


__all__ = ["Unit", "load_units", "summarise", "paired_tests", "holm", "analyse_directory"]
