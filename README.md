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

Requires Python 3.10+ (validated here on CPython 3.13.3, Windows 11). The core
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
fallbacks. `doctor` does not merely read the YAML: it encodes a probe vector
(checking dimensions, finite values and a non-zero norm), asks the generator for
a two-token completion, runs one NLI inference on a fixed premise, queries the
persistent ledger and reads the queue. `PROTECTED` requires every one of those
probes to pass; a configured model name alone can never earn it.
`probe_timeout_s` bounds the network probes (default 10 s), and no probe starts a
service or downloads weights. `UNPROTECTED` means no defence is active and the
run must not be labelled RAGLedger.

Run the deterministic end-to-end offline matrix (12 attacks × 11 policies × 1 seed,
132 cases; hashing embedder, stub generator, lexical verifier):

```bash
python scripts/run_chronological_eval.py --out runs/final_smoke
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
  --steps 12 --targets 2 --out runs/final_trust_stream
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
| `black_box` | No access to detector features |
| `white_box` | Feature-aware evasion |
| `sybil` | Corroboration split across synthetic sources |
| `coordinated` | Near-duplicate injection synchronised across sources |
| `adaptive` | Strategy changes after quarantine |

Policies: `no_defence`, `duplicate_filter`, `perplexity`, `suspicion_only`,
`corroboration_only`, `trust_threshold`, `trust_no_deferred`, adapted
`TrustRAG`, adapted `RobustRAG`, `always_on_loo`, and `RAGLedger`. The adapted
baselines are marked `published_method=False` and must not be described as
reproductions.

The research protocol, 12-attack definitions, statistical plan and real-model
commands are in [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md). The
offline result is in [`baseline_logs/research_smoke_full/`](baseline_logs/research_smoke_full/);
the IEEE draft is [`paper/main.tex`](paper/main.tex). The single-seed fixture
had zero detection precision/recall; a three-seed check found only weak,
isolated non-zero recall. It is negative/plumbing evidence only.

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
  probe_timeout_s: 10

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

Environment: Windows 11, CPython 3.13.3, no GPU, no `faiss-cpu`, `torch`,
`transformers` or `matplotlib`; `pyarrow 26.0.0` is installed for the Parquet
subset tests. Configuration `config/default.yaml` (hashing embeddings, numpy
index, stub LLM, lexical verifier, heuristic scorer).

### Tests

```bash
pytest -q
pytest -q --cov=trace_rag --cov-report=term-missing
pytest -q -W error::ResourceWarning
```

Result (offline; optional torch/faiss/matplotlib/nltk tests skip): **402 passed, 7 skipped**. Coverage: **81%** overall (6319 statements,
1214 missed). The ResourceWarning run is clean. Skips are optional backends:
`faiss-cpu` (2 tests), `torch` (1), `matplotlib` (3), `nltk` (1). New tests cover exact
Parquet subset construction, snapshot corruption/interruption/signature mixing,
completed-run identity reuse and rejection, preflight storage isolation,
signature-safe recovery and the notebook pin. Fresh logs for the tested SHA are
committed in [`baseline_logs/`](baseline_logs/); each records the command, exit
code, Python and OS versions, and the full tested Git SHA.

### Chronological smoke matrix

56/56 cases completed, no failures. RAGLedger results by attack. *Poison
exposure* is the share of eligible target attempts whose targeted poison reached
the generation context; *poison citation* is the share cited by the answer.
Neither is answer success. **Answer-level ASR is NOT MEASURED unless an evaluator-only answer file is supplied** because this
harness has no validated, attack-specific criterion for the attacker's intended
answer.

| Attack | Eligible attempts | Exposed (rate) | Cited (rate) | Precision | Recall | Clean FPR | LLM calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| entity_swap | 12 | 0 (0.000) | 0 (0.000) | 1.000 | 0.333 | 0.000 | 123 |
| negation | 12 | 4 (0.333) | 4 (0.333) | 0.000 | 0.000 | 0.000 | 123 |
| instruction_injection | 12 | 5 (0.417) | 3 (0.250) | 0.000 | 0.000 | 0.000 | 129 |
| slow_burn | 12 | 0 (0.000) | 0 (0.000) | 0.500 | 0.167 | 0.020 | 131 |
| hit_and_run | 12 | 2 (0.167) | 2 (0.167) | 0.500 | 0.167 | 0.026 | 115 |
| signal_evasion | 12 | 2 (0.167) | 2 (0.167) | 0.500 | 0.167 | 0.026 | 137 |
| framing | 12 | 4 (0.333) | 4 (0.333) | 0.000 | 0.000 | 0.000 | 123 |

Mean poison exposure across the seven attacks: RAGLedger **0.202**, no-defence
**0.929**; mean poison citation: RAGLedger **0.179**, no-defence **0.643**.
Mean clean FPR: RAGLedger **0.010**. The previously published "ASR 0.202" was
this exposure rate, not an answer-success rate; it is relabelled here. These
numbers show the harness and the lifecycle work; they do **not** establish that
RAGLedger detects most attacks. `negation`, `instruction_injection` and
`framing` still reach the context with the lexical verifier, and one correct
`newcomer` passage was quarantined in two cases. Treat this as a regression
baseline, not a result.

The full table, per-case metrics and streams are in
[`runs/final_smoke/`](runs/final_smoke/) (`results.json`, `results.csv`,
`summary.md`, `cases/`).

### Trust-stream smoke

On the bundled mini corpus (`--steps 12 --targets 2`), the lexical run had 6
eligible attack attempts, `poison_retrieved_rate=1.0`,
`poison_exposure_rate=0.0`, `poison_citation_rate=0.0`,
`clean_false_quarantine=0`, `poison_quarantined=0`, and 22 NEUTRAL
verifications. The poison was retrieved but withheld from the generation
context and never cited; it was **not** detected or quarantined. With the
lexical fallback the only contradicting copy of the claim was a near-duplicate
from the same family, so independence rules excluded it. This is an honest
limitation, not a success, and `answer_level_asr` is `NOT MEASURED`. See
[`runs/final_trust_stream/metrics.json`](runs/final_trust_stream/metrics.json).

### Demonstration of the full lifecycle

`tests/test_chronological_smoke.py` asserts the complete path on the real
pipeline: poison ingestion → MEDIUM/HIGH escalation → verification → trust
update → quarantine → later retrieval exclusion. `tests/test_trust_plan_gaps.py`
covers queue drain, background worker, administrator review, source/family
propagation and remediation. `tests/test_security_wiring.py` asserts the secure
default, the UNPROTECTED baseline report and every functional readiness probe
(embedding, generation, NLI, ledger, queue).

---

## Research readiness and real-model protocol

**No real-model experiment has been run here.** Two deliberately separate scopes
are defined in [`docs/REAL_MODEL_EVAL.md`](docs/REAL_MODEL_EVAL.md):

* **Integration pilot** — a small real-model run (5,000 passages, 20 queries) to
  prove the complete Contriever → FAISS → Llama → DeBERTa NLI → SQLite path.
  Its metrics are engineering validation, not research findings.
* **Research experiment** — the 200,000-passage / 500-query Natural Questions
  subset, not the full 2.68M corpus. It has not been executed.

### Preflight

`scripts/check_research_environment.py` inspects the host, packages, CUDA,
disk, configured endpoint, model cache, persistent ledger, dataset files and
configuration. It never downloads weights or starts a service. Static success
is only `DEGRADED`; `READY` requires the functional doctor to report
`PROTECTED`:

```bash
python scripts/check_research_environment.py --config config/nq_gpu.yaml
python scripts/check_research_environment.py --config config/nq_gpu.yaml \
  --functional --preflight-storage-root /tmp/ragledger-preflight \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv
```

Current host result: **BLOCKED** — `torch`, `transformers` and `faiss` are
absent, CUDA is unavailable, Contriever is not cached, and no generator is
served at `localhost:8000`. That is an accurate environment result, not a model
result.

### Integration pilot

Prepare a small subset (the downloader first fetches the BEIR source corpus;
do not do this on a constrained disk), start vLLM, then run with the explicit
fail-closed flag:

```bash
python scripts/download_data.py --dataset nq --out data/nq_pilot \
  --subset 5000 --queries 20
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python scripts/check_research_environment.py --config config/pilot_gpu.yaml \
  --functional --corpus data/nq_pilot/corpus_subset.parquet \
  --queries data/nq_pilot/queries_subset.jsonl --qrels data/nq_pilot/qrels.tsv
python scripts/run_trust_stream.py --config config/pilot_gpu.yaml \
  --corpus data/nq_pilot/corpus_subset.parquet \
  --queries data/nq_pilot/queries_subset.jsonl --qrels data/nq_pilot/qrels.tsv \
  --run-purpose pilot --sample-mode zipf \
  --limit 5000 --steps 12 --targets 2 --out runs/pilot/trust_stream \
  --require-protected
```

### Research experiment

```bash
python scripts/download_data.py --dataset nq --out data/nq \
  --subset 200000 --queries 500
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python scripts/check_research_environment.py --config config/nq_gpu.yaml \
  --functional --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv
python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
  --corpus data/nq/corpus_subset.parquet \
  --queries data/nq/queries_subset.jsonl --qrels data/nq/qrels.tsv \
  --run-purpose research --sample-mode no-repeat \
  --steps 500 --targets 5 --out runs/nq/trust_stream
```

For a non-stub generation backend the runner now refuses to proceed unless the
functional doctor is `PROTECTED`, the verifier is NLI and the trust ledger is
persistent. `--require-protected` additionally enforces that gate for any
backend. `run_manifest.json` records the commit, dataset hashes, resolved
configuration, model backends/revisions/precision/quantisation, seed, hardware,
ledger path, indexed passages, query pool, stream steps and distinct queries. A
500-step Zipf run is labelled `research_lifecycle` and is never 500 unique
queries; the primary research run uses `--sample-mode no-repeat`. The runner
writes exposure/citation and document-level precision/recall/F1/FPR;
answer-level ASR remains **NOT MEASURED** unless `--answers` supplies evaluator-only gold answers; this runner is not the full attack
× baseline matrix.

---

## Known limitations

* **No real-model or full-dataset run has been performed here.** No poison
  exposure, citation, answer-level ASR, detection or cost claim in this
  repository comes from Contriever, an 8B LLM, DeBERTa NLI or full Natural
  Questions.
* The lexical verifier cannot adjudicate a one-against-one contradiction
  between equally trusted sources. It withholds both rather than choosing, but
  repeated conflict can still quarantine a correct passage; 1–2.6% clean FPR
  was observed in three smoke cases.
* `negation`, `instruction_injection` and `framing` evade the cheap signals and
  the lexical verifier often enough to leave poison exposure at 0.33–0.42.
  These are fixture-era numbers; answer-level ASR is now measured by the real-data
  matrix (a string-presence proxy, see docs/AUDIT_RESPONSE.md) but has not yet been run on a GPU.
* The smoke corpus is trivially separable; all signals can separate the toy
  poison. It is unsuitable for measuring detection quality.
* Optional real backends (`faiss`, `torch`, `transformers`, `pyarrow`,
  `matplotlib`) are not installed in this environment; their tests are skipped
  or their code paths unexercised.
* The research preflight is **BLOCKED** on this host. The integration pilot and
  the 200K/500 research experiment cannot run until the documented GPU, model
  cache, generator endpoint and Python dependencies exist.
* The `run_trust_stream.py` mini run does not train the suspicion scorer; it
  uses the heuristic thresholds. Real runs should train and validate the scorer
  on labelled rows first.
* `scripts/run_poisonedrag_official.py` is the real-data matrix runner (published
  PoisonedRAG poison on BEIR NQ). It is tested offline only; no GPU result exists yet.
  `docs/REAL_RESULTS_PLAN.md` says what may and may not be claimed.
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
