# RAGLedger

**Adaptive trust-based framework for detecting and preventing knowledge-base
poisoning in retrieval-augmented generation (RAG) systems.**

RAGLedger places a security layer between retrieval and generation. It scores
retrieved passages with six cheap signals, verifies suspicious claims against
independent evidence, records trust at document, near-duplicate-family and
source level, quarantines content that accumulates refutations, and flags past
answers that relied on content quarantined later.

This repository is the integrated system, not a proposal: all three
workstreams (ingestion/detection/generation, trust/verification/policy, and
attacks/baselines/evaluation) now run through one pipeline.

> **Scope of the results in this repository.** The included smoke results use a
> tiny synthetic corpus, the deterministic hashing embedder, a stub LLM and the
> lexical verifier. They validate plumbing and lifecycle behaviour. They are
> **not** benchmark numbers and must not be reported as research results. The
> real-model Natural Questions protocol is documented in
> [`docs/REAL_MODEL_EVAL.md`](docs/REAL_MODEL_EVAL.md) and has **not** been run
> in this environment.

---

## Quick start (offline, no model downloads)

Requires Python 3.10+ (validated here on CPython 3.13.3, Windows). The core
install has no GPU, network or model dependency.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
pytest -q
```

Check that the security controls are actually attached before running anything:

```bash
trace-rag doctor
# security=DEGRADED ledger=TrustLedger verifier=CorroborationVerifier(lexical)
# policy=TrustPolicy queue=on embedder=hashing llm=stub
```

`DEGRADED` is deliberate: the trust ledger, verifier, policy and queue are
active, but the offline embedder/LLM and lexical verifier are documented
fallbacks. It becomes `PROTECTED` only with the NLI verifier, persistent ledger,
queue and real backends. `UNPROTECTED` means no defence is active and the run
must not be labelled RAGLedger.

Run the deterministic end-to-end smoke matrix (7 attacks × 8 baselines × 1 seed,
56 cases):

```bash
python scripts/run_chronological_eval.py --out runs/smoke_matrix
```

Run the component demos:

```bash
python scripts/demo_end_to_end.py --root runs/demo
python scripts/demo_person_b.py --root runs/demo_b
```

`demo_end_to_end.py` is the explicitly **unprotected Person A demo**: it shows
ingestion, signals, scoring and remediation without the trust policy.
`demo_person_b.py` exercises the protected stack, including queued verification,
quarantine and retroactive remediation.

Run the trust stream on the bundled mini corpus:

```bash
python scripts/run_trust_stream.py \
  --config config/default.yaml \
  --corpus examples/mini_corpus.jsonl \
  --queries examples/mini_questions.json \
  --steps 12 --targets 2 --out runs/trust_stream_mini
```

### What the smoke matrix covers

| Attack family | Injected behaviour |
|---|---|
| `entity_swap` | Correct answer replaced by a false named entity |
| `negation` | Claim negated |
| `instruction_injection` | In-context instruction telling the model to ignore policy |
| `framing` | A denial of the true claim, without obvious attack markers |
| `slow_burn` | Benign-looking history from the attacker's source before poison |
| `hit_and_run` | One-shot injection from a fresh source |
| `signal_evasion` | Query echo removed / text perturbed to evade cheap signals |

Baselines: `no_defence`, `duplicate_filter`, `trust_threshold`, `perplexity`,
`TrustRAG`, `RobustRAG`, `always_on_loo`, and `RAGLedger`.

---

## Architecture

```
documents
   │
   ▼
Ingestion + provenance (SQLite: sources, documents, chunks, families, hashes)
   │
   ▼
Embedding + index (hashing/numpy offline; Contriever/FAISS for real runs)
   │
   ▼
Trust-weighted retrieval  score = similarity × max(t_eff, floor)^λ
   │
   ▼
Six cheap signals S1..S6 → calibrated suspicion scorer → LOW / MEDIUM / HIGH
   │
   ▼
TrustPolicy ── LOW ──────────────────────────────► use
   │            MEDIUM → verifier ─ SUPPORT ─────► use
   │                               REFUTE ──────► exclude + trust penalty
   │                               NEUTRAL ─────► withhold (fail closed)
   │            HIGH ─────────────► exclude + queue for off-path verification
   │            QUARANTINED ──────► never retrieved or generated from
   ▼
Grounded generation (citations required; abstain when evidence mass is low)
   │
   ▼
Answer provenance log ──► retroactive remediation when a passage is quarantined
   │
   ▼
Trust ledger (doc / family / source, decay, bursts, observations)
   │
   ▼
Off-path verification queue → refute → quarantine → flag earlier answers
```

### Module map

| Layer | Module | Responsibility |
|---|---|---|
| Contracts | `contracts.py` | Versioned dataclasses and protocols shared by all workstreams |
| Configuration | `config.py` | Typed pydantic config; every threshold lives here |
| Time | `utils/clock.py` | Wall clock or deterministic logical clock for streams |
| Ingestion | `ingestion/` | Parsers, ~100-word chunking, provenance, family detection |
| Embedding/index | `embeddings/`, `index/` | Hashing + HuggingFace embedders; numpy + FAISS indexes |
| Retrieval | `retrieval/retriever.py` | Candidate pool and trust-weighted top-k; blocked docs removed |
| Detection | `detection/` | S1–S6 signals, logistic scorer, calibration, leakage-guarded training |
| Verification | `trust/verifier.py` | Claim extraction, influence gate, independent corroboration, NLI/lexical |
| Trust state | `trust/ledger.py` | Doc/family/source trust, observations, decay, quarantine, admin review |
| Policy | `trust/policy.py` | Band escalation, fail-closed handling, quarantine and remediation |
| Queue | `trust/queue.py` | Bounded off-path verification queue and background worker |
| Wiring | `security.py` | Attaches the stack by default and reports PROTECTED/DEGRADED/UNPROTECTED |
| Generation | `generation/` | Role-separated prompts, citations, abstention, LLM backends |
| Provenance | `provenance/` | Answer log and retroactive remediation |
| Attacks | `attacks/` | PoisonedRAG-style generators, adaptive families, stream contracts |
| Baselines | `baselines/` | Eight defences over one policy interface |
| Evaluation | `evaluation/` | Chronological runner, metrics, statistics, experiment matrix |
| CLI | `cli.py` | `ingest`, `ingest-beir`, `index`, `query`, `smoke`, `doctor`, ... |

### Security state machine

| Input state | Action for the current answer | Trust consequence |
|---|---|---|
| LOW | Use | None |
| MEDIUM + SUPPORT | Use | Positive observation (deduplicated per query) |
| MEDIUM + REFUTE | Exclude | Negative observation; may quarantine |
| MEDIUM + NEUTRAL | Withhold | No trust change (unresolved is not support) |
| MEDIUM, no verifier | Withhold (default) | None; research-only `fail_open_without_verifier` exists |
| HIGH | Exclude and queue | Passive penalty, then off-path verification |
| QUARANTINED / REJECTED | Never retrieved or generated from | Only administrator review recovers it |

Inline and queued verification share the same ledger update and remediation
path. A queued refutation therefore quarantines the passage and flags earlier
answers that used it.

### Shared data contracts

The stream boundary accepts legacy `{t, qid, text}` rows and canonical
`{step, query_id, query}` rows, but only through the adapters in
`attacks/stream.py`:

```python
from trace_rag.attacks.stream import as_query_row, as_ingestion_row

query = as_query_row({"t": 3, "qid": "q3", "text": "Who?"})
# {"step": 3, "query_id": "q3", "query": "Who?"}
```

Required fields are validated and raise `StreamContractError`. Baselines read
trust through `doc.trust.t_eff` (the effective value used by retrieval), never
an ad-hoc `trust` attribute. See [`docs/INTERFACES.md`](docs/INTERFACES.md) for
the object-level contract (that document is the original Person A handover;
the code and this README take precedence where they differ).

---

## Configuration

Two configurations are supported:

| File | Use |
|---|---|
| `config/default.yaml` | Offline development: hashing embedder, numpy index, stub LLM, lexical verifier |
| `config/nq_gpu.yaml` | One-GPU Natural Questions run: Contriever, FAISS flat, vLLM/Ollama, NLI verifier |
| `config/full_nq.yaml` | Full corpus: Contriever, FAISS IVF-PQ, vLLM, NLI verifier |

Security is on by default:

```yaml
security:
  enabled: true
  verifier: lexical         # lexical | nli | auto
  max_corroboration: 3
  queue_size: 1000
  drain_queue: true
  fail_open_without_verifier: false
  medium_neutral_action: exclude
  min_refute_topical_overlap: 1
  min_refute_topic_coverage: 0.50
  min_refute_query_coverage: 0.60
  support_threshold: 0.50
  refute_threshold: 0.10
  max_source_mass: 0.6
  counterfactual_influence: true

trust:
  quarantine_refutations: 2
  reject_refutations: 3
  burst_window_seconds: 60
  burst_min_docs: 4
  cold_start_influence: 0.25
```

Overrides work before or after the subcommand:

```bash
trace-rag --config config/default.yaml query "Who designed the Eiffel Tower?" \
  --set generation.backend=stub --set security.verifier=lexical
```

### Degraded modes (explicit, not silent)

* `verifier=lexical` when the NLI model is unavailable or not requested.
* `embedding=hashing` and `generation=stub` for offline tests.
* In-memory trust ledger if no path is configured.
* `security.enabled=false` or `secure=False` creates an **UNPROTECTED** baseline;
  the CLI warns and `doctor` exits non-zero.

---

## Verified results (smoke scope only)

Environment: Windows, CPython 3.13.3, no GPU, no `faiss-cpu`, `torch`,
`transformers`, `pyarrow` or `matplotlib`. Configuration
`config/default.yaml` (hashing embeddings, numpy index, stub LLM, lexical
verifier, heuristic scorer).

### Tests

```bash
pytest -q
pytest -q --cov=trace_rag --cov-report=term-missing
```

Result: **235 passed, 6 skipped**. Coverage: **80%** overall. Skips are optional backends:
`faiss-cpu` (2 tests), `torch` (1), `matplotlib` (3). Full logs:
[`baseline_logs/`](baseline_logs/).

### Chronological smoke matrix

56/56 cases completed, no failures. RAGLedger results by attack (ASR lower is
better; precision/recall are document-level detection):

| Attack | ASR | Precision | Recall | Clean FPR | LLM calls |
|---|---:|---:|---:|---:|---:|
| entity_swap | 0.000 | 1.000 | 0.333 | 0.000 | 123 |
| negation | 0.333 | 0.000 | 0.000 | 0.000 | 123 |
| instruction_injection | 0.417 | 0.000 | 0.000 | 0.000 | 129 |
| slow_burn | 0.000 | 0.500 | 0.167 | 0.020 | 131 |
| hit_and_run | 0.167 | 0.500 | 0.167 | 0.026 | 115 |
| signal_evasion | 0.167 | 0.500 | 0.167 | 0.026 | 137 |
| framing | 0.333 | 0.000 | 0.000 | 0.000 | 123 |

Mean ASR across the seven attacks: RAGLedger **0.202**, no-defence **0.929**.
Mean clean FPR: RAGLedger **0.010**. These numbers show the harness and the
lifecycle work; they do **not** establish that RAGLedger detects most attacks.
`negation`, `instruction_injection` and `framing` are still largely missed with
the lexical verifier, and one correct `newcomer` passage was quarantined in two
cases. Treat this as a regression baseline, not a result.

The full table, per-case metrics and streams are in
[`runs/smoke_matrix/`](runs/smoke_matrix/) (`results.json`, `results.csv`,
`summary.md`, `cases/`).

### Trust-stream smoke

On the bundled mini corpus (`--steps 12 --targets 2`), the lexical run had
`poison_retrieved_rate=1.0`, `poison_citation_rate=0.0`,
`clean_false_quarantine=0`, `poison_quarantined=0`, and 22 NEUTRAL
verifications. The poison was never cited, but it was not quarantined: with the
lexical fallback the only contradicting copy of the claim was a near-duplicate
from the same family, so independence rules excluded it. This is an honest
limitation, not a success. See
[`runs/trust_stream_check/metrics.json`](runs/trust_stream_check/metrics.json).

### Demonstration of the full lifecycle

`tests/test_chronological_smoke.py` asserts the complete path on the real
pipeline: poison ingestion → MEDIUM/HIGH escalation → verification → trust
update → quarantine → later retrieval exclusion. `tests/test_trust_plan_gaps.py`
covers queue drain, background worker, administrator review, source/family
propagation and remediation. `tests/test_security_wiring.py` asserts the secure
default and the UNPROTECTED baseline report.

---

## Real-model Natural Questions experiment

Not run here. The exact prerequisites and commands are in
[`docs/REAL_MODEL_EVAL.md`](docs/REAL_MODEL_EVAL.md). Short version:

```bash
pip install -e ".[all,data]"
python scripts/download_data.py --dataset nq --out data/nq --subset 200000 --queries 500
trace-rag --config config/nq_gpu.yaml ingest-beir --corpus data/nq/corpus_subset.parquet
trace-rag --config config/nq_gpu.yaml index
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl \
  --qrels data/nq/qrels.tsv --out runs/nq/trust_stream
```

The NLI verifier downloads `cross-encoder/nli-deberta-v3-base` on first use.
A real run is only reportable when `trace-rag doctor` says `PROTECTED`, the
verifier mode is `nli`, and the run has a persistent trust database. Dataset
licensing and citation requirements are in `docs/REAL_MODEL_EVAL.md`.

---

## Known limitations

* **No real-model or full-dataset run has been performed here.** No ASR,
  detection or cost claim in this repository comes from Contriever, an 8B LLM,
  DeBERTa NLI or full Natural Questions.
* The lexical verifier cannot adjudicate a one-against-one contradiction
  between equally trusted sources. It withholds both rather than choosing, but
  repeated conflict can still quarantine a correct passage; 1–2.6% clean FPR
  was observed in three smoke cases.
* `negation`, `instruction_injection` and `framing` evade the cheap signals and
  the lexical verifier often enough to leave ASR above zero (0.33–0.42).
* The smoke corpus is trivially separable; all signals can separate the toy
  poison. It is unsuitable for measuring detection quality.
* Optional real backends (`faiss`, `torch`, `transformers`, `pyarrow`,
  `matplotlib`) are not installed in this environment; their tests are skipped
  or their code paths unexercised.
* The `run_trust_stream.py` mini run does not train the suspicion scorer; it
  uses the heuristic thresholds. Real runs should train and validate the scorer
  on labelled rows first.
* There is no real-NQ attack/baseline matrix runner yet. The chronological
  matrix is fixture-only; `run_trust_stream.py` is the available real-data
  end-to-end path.
* The repository still contains historical Person A/B handover documents whose
  statements may predate the integration. Code, README and the engineering
  report are authoritative.

---

## Repository layout

```
src/trace_rag/
  contracts.py config.py pipeline.py security.py cli.py
  ingestion/ embeddings/ index/ retrieval/ detection/
  trust/ generation/ provenance/
  attacks/ baselines/ evaluation/
  utils/
config/       default.yaml, nq_gpu.yaml, full_nq.yaml
examples/     mini corpus, late clean passages, questions, poison fixture
scripts/      demos, trust stream, chronological matrix, setup, profiling, data download
tests/        unit, integration, security, chronological smoke, regression
docs/         engineering report, real-model protocol, interfaces, handovers
baseline_logs/ captured output from the verification runs
runs/         generated smoke artefacts (not source)
```

## Documentation

* [`docs/ENGINEERING_REPORT.md`](docs/ENGINEERING_REPORT.md) — audit, defect
  register, file-by-file changes, test/evaluation report, limitations,
  reproduction commands.
* [`docs/REAL_MODEL_EVAL.md`](docs/REAL_MODEL_EVAL.md) — prerequisites and
  exact commands for a GPU Natural Questions run.
* [`docs/INTERFACES.md`](docs/INTERFACES.md) — object contracts and how the
  pipeline is wired (historical handover; code wins).
* [`docs/RUNNING_ON_GPU.md`](docs/RUNNING_ON_GPU.md) — CUDA/vLLM/Ollama setup.
* [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — earlier Person A defect
  history, retained for context.

## Licence and data

Project code is MIT (see `LICENSE`). The bundled `examples/` corpus is
synthetic and generated for tests. Natural Questions / BEIR data is downloaded
separately by `scripts/download_data.py`; check the dataset's own licence and
citation terms before redistributing it. Do not commit downloaded corpora or
model weights.
