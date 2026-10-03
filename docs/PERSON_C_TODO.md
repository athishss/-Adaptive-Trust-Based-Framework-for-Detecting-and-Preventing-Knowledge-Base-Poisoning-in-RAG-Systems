# What Person C needs to fix

Audit of `src/trace_rag/{attacks,baselines,evaluation}` at commit `9a2f88f`, done
by reading the code, driving it, and measuring coverage. Every claim below is
reproduced, not inferred. If something here disagrees with the code, the code
wins and this file is a bug.

## Status in one line

These three packages are **scaffolding, not a harness**: 12 files, 316
statements, **0 covered by any test**, and **no external caller anywhere in
`src/`, `scripts/` or `tests/`** for `run_stream`, `build_experiment_matrix`,
`EvaluationResult`, `always_on_loo`, `BASELINE_NAMES`, `bootstrap_ci`,
`mcnemar_exact`, `zipf_queries`, `build_query_stream`, `slow_burn` or
`signal_evasion`.

**Consequence: no research number can be produced from this repo yet.** ASR,
detection delay, exposure window and the security/cost trade-off all run
through this code.

---

## P0 - the C4 harness does not connect to itself

`evaluation/chronological.py::run_stream` is the runner. It reads dict events
and dict queries. The two adapters that are supposed to feed it do not produce
those shapes, and there is no adapter for the ingestion side at all.

Reproduced exactly:

| Fed in | Result |
|---|---|
| `evaluation.integration.to_pipeline_queries(...)` -> `run_stream` | `KeyError: 'step'` |
| `attacks.stream.build_ingestion_schedule(...)` -> `run_stream` | `TypeError: 'StreamIngestion' object is not subscriptable` |
| Hand-built dicts in the shape `run_stream` wants | works, reaches `pipeline.answer()` |

The key mismatch, exact:

```
to_pipeline_queries emits : {'text', 'qid', 't'}
run_stream reads          : query['query'], query.get('query_id'), query['step']
```

`run_stream` also needs, per ingestion event:
`step`, `doc_id`, `source_id`, `text`, and optionally `title`, `origin`,
`is_poison`, `family_id`. `StreamIngestion` already carries almost exactly
these fields - the adapter is small, it just does not exist.

**Fix:** write the two adapters (query side and ingestion side) and add a smoke
test that runs a handful of steps end to end over `examples/mini_corpus.jsonl`
plus `examples/mini_poison.jsonl`. Until that test exists, this keeps breaking.

## P0 - nothing runs the experiment matrix

`evaluation/runner.py` defines `EvaluationCase`, `EvaluationResult` and
`build_experiment_matrix`, and nothing calls any of them. There is no function
that takes an `EvaluationCase` and produces an `EvaluationResult` - the
attack x baseline x seed matrix is data with no executor.

`EvaluationResult` also has no `to_dict()`, and this repo's contract
(`contracts.py`) requires every payload dataclass to serialise to JSON
primitives, so results cannot be written to a `runs/` artifact as they stand.

**Fix:** a runner that executes a case (build stream -> run it -> compute
metrics) and returns a result, plus `to_dict()` on the result types.

## P1 - two of the six advertised baselines do not exist

`baselines/policies.py::BASELINE_NAMES` lists six:

```
no_defence, perplexity, duplicate_filter, TrustRAG, RobustRAG, always_on_loo
```

Implemented functions: `no_defence`, `duplicate_filter`, `perplexity_filter`,
`trust_threshold_filter`, `always_on_loo`. `TrustRAG` and `RobustRAG` appear
**only as those two strings** - there is no implementation in the repository
(`grep -rn "TrustRAG\|RobustRAG" src` returns the tuple entries and a comment
in `detection/signals.py`).

Also: `trust_threshold_filter` exists but is not in `BASELINE_NAMES`, so the
list and the code disagree in both directions.

**Fix:** implement TrustRAG and RobustRAG, or remove them from the list and
state in the paper that they were not run. Do not ship a baseline table whose
rows are named but unimplemented.

## P1 - the attack generators violate the shared contract

- `PoisonedDocument`, `StreamIngestion` and `AttackEvent` are frozen dataclasses
  with **no `to_dict()`**, against the explicit rule in `contracts.py`
  ("every payload object is a frozen dataclass with `to_dict()` producing
  JSON-serialisable primitives only"). They cannot be logged into experiment
  artifacts.
- `PoisonedDocument.metadata` is annotated `Mapping[str, object]` but defaults
  to `None`. Only `make_poisoned_document` normalises it to `{}`; construct the
  dataclass directly and `dict(doc.metadata)` raises. It should be
  `Optional[Mapping[str, object]] = None` with `field(default_factory=dict)`.
- `attacks/__init__.py` is **empty (0 bytes)** while `baselines/__init__.py`,
  `evaluation/__init__.py` and `trust/__init__.py` all re-export their public
  API. So `import trace_rag.attacks.poisoned_rag` works but
  `from trace_rag.attacks import entity_swap` raises `ImportError`. Pick one
  convention and make the four packages match.

## P1 - three generators do not do what their names claim

- `negation(text, claim)`: if the claim is not verbatim in the text it returns
  `f"Not true: {claim}. {text}"`, i.e. it **keeps the true claim** and prepends
  a denial. That is a self-contradictory passage, not a corruption - and the
  corroboration verifier can read the retained true sentence as *support*.
- `entity_swap(text, original, replacement)`: unbounded, case-sensitive
  `str.replace`, so it rewrites **every** occurrence. PoisonedRAG's
  construction needs one targeted replacement.
- `slow_burn(payloads, warmup_steps=5)`: `warmup_steps` only **offsets the step
  counter**. It emits no benign warm-up documents from that source, so no
  reputation is ever built and the attack does not test `TrustLedger` alpha
  accumulation - which is the entire point of C3.
- `framing(payloads, ...)`: emits single-document events with no pooling of
  legitimate evidence, so it cannot make true documents look isolated, which is
  what its own docstring claims it is for.

`hit_and_run` and `signal_evasion` are thin wrappers that differ only in the
`attack_type` string and the source naming; neither generates any attack text.

## P2 - the metrics quietly report the optimistic answer

`evaluation/metrics.py` returns `0.0` when the denominator is zero:

```python
def attack_success_rate(successes, attacks):
    if attacks <= 0: return 0.0        # reads as "no attack succeeded"
```

Same for `false_positive_rate`, `remediation_recall`. An empty or
mis-matched experiment therefore reports a perfect security result instead of
failing. In a security paper that is the wrong direction to fail.
**Fix:** raise, or return `None`, and make the caller decide.

Two smaller notes: `retrieval_metrics` returns precision 0.0 when the relevant
set is empty even if something was retrieved (defensible, but document it), and
`always_on_loo` compares `result.outcome.value == "REFUTE"` as a string where
the enum is available.

## P2 - the query schedule and the ingestion schedule are computed independently

`build_query_stream` marks a step as a target query using
`gradual_schedule(total_steps, poison_start, poison_every)`
(= `range(poison_start, total_steps, poison_every)`), while
`build_ingestion_schedule` places poison at
`poison_start + i * poison_every`. Nothing enforces that a poison document is
ingested **before** the target query that should be affected by it.

Detection delay and exposure window are both measured in stream steps, so if
the ordering is not guaranteed those two numbers are not well defined.
**Fix:** derive both schedules from one ordering, and assert in the harness
that every target query step is greater than the ingestion step of the poison
it targets.

## What Person B changed that you must account for

These landed after your commit, in `trust/verifier.py`:

1. **Refutation now requires topical overlap** with the claim. The NLI model
   returns `contradiction` at confidence 1.0 for premises that never mention
   the claim, so a single off-topic passage used to REFUTE an honest one
   (`refute_mass` 0.50 against `min_mass` 0.10). If you measure
   false-quarantine rates, expect this to matter.
2. **`CorroborationVerifier.mode`** is now a property returning `"nli"` or
   `"lexical"`, and falling back to lexical logs a **warning**. Record
   `verifier.mode` on every `EvaluationCase` - otherwise a run without the NLI
   model available is not comparable with one that had it, and the summary
   table will silently mix methods.
3. **Known limitation: `SUPPORT` is near-unreachable.** The leave-one-out
   influence proxy strips query tokens, shrinking the claim's content words to
   the answer tokens, so a passage that actually entails the claim is judged
   redundant and the verifier returns `NEUTRAL` before NLI runs. Interpret
   "verification rate" and "support rate" metrics accordingly.

## Coverage requirement

All 12 files are at **0%**. Before any number from this code goes in a paper,
each of these needs the same standard Person A and Person B were held to - a
test that fails if the behaviour breaks:

```
attacks/adaptive.py          24 stmts    0%
attacks/poisoned_rag.py      35 stmts    0%
attacks/stream.py            64 stmts    0%
baselines/policies.py        56 stmts    0%
evaluation/chronological.py  31 stmts    0%
evaluation/integration.py     5 stmts    0%
evaluation/metrics.py        45 stmts    0%
evaluation/runner.py         28 stmts    0%
evaluation/statistics.py     24 stmts    0%
```

Person A: 92.0% (2703 stmts). Person B: 95.4% (498 stmts). Repo: 84.2%.
