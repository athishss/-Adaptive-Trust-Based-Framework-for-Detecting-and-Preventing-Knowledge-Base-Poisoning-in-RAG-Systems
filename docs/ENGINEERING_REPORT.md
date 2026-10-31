# RAGLedger engineering report

**Repository:** `RAGLedger` (TRACE-RAG adaptive trust framework)
**Audit date:** 2026-10-31
**Implementation commit under test:** `58de432c3124cc7cf81dcbfc4cc169b91c00af08`
**Environment:** Windows 11 (10.0.26200), CPython 3.13.3, AMD64, no GPU; core
dependencies plus `pyarrow 26.0.0` installed for the Parquet subset tests
(numpy 2.5.4, scikit-learn 1.9.1, pydantic 2.14.1, PyYAML 6.0.3, pytest 9.1.1,
pytest-cov 7.1.0). Optional backends `faiss-cpu`, `torch`, `transformers` and
`matplotlib` are **not installed**.
**Scope:** Colab preflight storage isolation, exact subset construction,
snapshot/recovery hardening, notebook pin correctness, documentation
reconciliation and offline validation. No security-policy thresholds changed.

---

## A. Executive summary

Earlier passes fixed the ASR/exposure label, functional readiness probes, SQLite
lifecycle and prompt-role separation. This pass audited the Colab handoff and
found four blockers that could invalidate a run before it started:

* the functional preflight constructed a real pipeline against the experiment's
  `storage.root`, leaving SQLite files in a fresh run directory; the runner then
  refused that directory as "partial artifacts";
* subset construction had no duplicate or blank identifier guard, malformed
  qrels/query rows could be silently dropped, and the tests only checked a loose
  size range;
* snapshot replacement was in place and metadata was not tied to a run
  signature, so an interrupted restore or a different experiment's snapshot
  could be mixed into the active run;
* the notebook pinned a stale implementation commit and downloaded the
  multi-gigabyte NQ corpus implicitly.

After this pass:

| Check | Result |
|---|---|
| Full test suite at the tested commit | **340 passed, 6 skipped** |
| Coverage | **81%** (6299 statements, 1212 missed) |
| Python compilation | `compileall` exit 0 |
| ResourceWarnings | **0** with `-W error::ResourceWarning` |
| Subset tests (direct Parquet) | 34 tests: exact size, gold coverage, seeds, duplicates, invalid rows |
| Snapshot/recovery tests | 16 tests: corruption, interruption, signature mixing, atomic restore |
| Preflight isolation tests | 12 tests, including a real offline doctor into a fresh directory |
| Notebook | 30 cells, 15 code cells, all parse, no saved outputs, pin is an ancestor of HEAD |
| Offline mini stream with `--backup-dir` | exit 0; 4/4 steps; snapshot chain written |
| Answer-level ASR | **NOT MEASURED** (no validated attack-specific answer criterion) |
| Real-model NQ run / Colab execution | **not run** |

The system remains reproducible at smoke scale. It is **not** a validated
benchmark result: the lexical verifier still misses several attack families,
one clean passage is falsely quarantined in two smoke cases, and no real-model
or full-dataset experiment has been executed.

### Research-environment preflight

`scripts/check_research_environment.py` is the non-invasive gate for a
real-model run. It reports host/CPU/memory/disk, torch/CUDA/VRAM, the optional
scientific packages, the configured generator endpoint and served model, the
Hugging Face cache, the persistent SQLite ledger, dataset files and the resolved
config. It never downloads weights or starts a service. Static checks alone can
only produce `DEGRADED`; `READY` requires `--functional` and a `PROTECTED`
functional doctor. Current NQ result on this host: **BLOCKED** — torch,
transformers and faiss are absent, CUDA is unavailable,
`facebook/contriever` is not cached, and `localhost:8000` is not serving the
configured Llama model.

The functional doctor builds a real `PersonAPipeline`, so this pass isolated its
storage. It always receives a unique disposable probe root (from
`--preflight-storage-root`, or a system temporary directory), copies the
resolved config with model/security/generation settings unchanged, rebases
absolute or `..` storage paths to safe basenames, records
`functional_doctor.storage_isolation`, and deletes the probe tree. A regression
runs the real CLI doctor with stub/lexical config and confirms a fresh
experiment directory remains empty, so
`check_output_state(fresh, resume=False)` accepts it. This was the first Colab
blocker: before the fix, the doctor created `answers.sqlite3`,
`provenance.sqlite3` and `trust.sqlite3` in the run root and the runner then
refused the directory.

The trust-stream runner fails closed before ingestion for any non-stub
generation backend (or with `--require-protected`) unless the doctor is
`PROTECTED`, `verifier_mode == "nli"` and `trust_persistent` is true. A
successful real run writes `run_manifest.json` with the commit, dataset hashes,
config, models, seed, hardware, ledger path and counts. The offline mini
manifest explicitly says `run_kind: offline_smoke`.

### Exact subsets and snapshot recovery (this pass)

`src/trace_rag/ingestion/subset.py` now streams the Parquet corpus with bounded
memory, writes through a temporary file plus `os.replace`, rejects duplicate or
blank document/query IDs, rejects missing gold passages and insufficient rows,
and verifies exact output count, unique IDs, query membership and 100% gold
coverage. `read_qrels_set` accepts header or headerless BEIR TSV but rejects
malformed rows and non-finite scores; `write_queries_exact` rejects duplicate,
blank, missing or text-less queries. Tests in `tests/test_ingestion_subset.py`
build real Parquet fixtures for exact size, determinism, seeds, all-gold,
missing gold, insufficient rows, duplicate IDs, invalid queries and both qrels
layouts.

`src/trace_rag/snapshots.py` now stages snapshot copies, validates manifest
signature/hash, SQLite integrity, stream/checkpoint agreement and the complete
index file set, records `run_signature` in metadata, refuses to restore a
snapshot from another experiment signature, installs `checkpoint.json` last as
the commit marker, and keeps a recoverable `.<name>.previous` index directory.
Interrupted staging is cleaned or refused. `tests/test_snapshots.py` covers
corruption, interruption, signature mixing and restore atomicity.

---

## B. File-by-file change log

### Configuration and scripts

| File | Change |
|---|---|
| `config/default.yaml` | Added `time`, `security` and `trust` sections; security enabled by default with lexical verifier, fail-closed MEDIUM handling, queue and evidence thresholds. |
| `config/nq_gpu.yaml` | Added the same security/trust surface with `verifier: nli`, persistent ledger, queue and production quarantine thresholds. |
| `config/full_nq.yaml` | Same real-model security configuration for the full-corpus IVF-PQ setup. |
| `config/pilot_gpu.yaml` | **New:** explicit 5K/20-query integration-pilot configuration under `runs/pilot`; it keeps the research model identifiers and is not a results configuration. |
| `scripts/check_research_environment.py` | **New:** READY/DEGRADED/BLOCKED preflight for Python, OS, CPU/RAM/disk, CUDA/VRAM, scientific packages, configured endpoint and served model, HF cache, persistent ledger, dataset and config. No downloads or service starts. **This pass:** `--preflight-storage-root`, `isolate_storage_config()`, non-creating ledger diagnostics and `functional_doctor.storage_isolation`; the real doctor never writes into the experiment root. |
| `config.py` | Added `probe_timeout_s` (default 10 s, bounded 0–120 s) for the functional readiness probes. **This pass:** non-mutating `resolve_path()` and `trust_db_path(create_parent=False)` so read-only diagnostics do not create experiment directories. |
| `scripts/demo_end_to_end.py` | Explicitly builds an **UNPROTECTED** Person A demo (`secure=False`) and prints that mode. |
| `scripts/demo_person_b.py` | Reuses the attached security stack; scripted verifier exercises queue → refute → quarantine → remediation. |
| `scripts/run_trust_stream.py` | Reuses the attached stack; figures optional; metrics state answer-level ASR is **NOT MEASURED**. **This pass:** fail-closed gate for real backends (`PROTECTED` + NLI + persistent ledger) before ingestion; `run_manifest.json` provenance; document-level precision/recall/F1/FPR and detection delay. |
| `scripts/run_chronological_eval.py` | **New** supported matrix entry point (7 attacks × 8 baselines, deterministic fixture, machine-readable outputs). |

### Core integration and time model

| File | Change |
|---|---|
| `utils/clock.py` | **New** `Clock` protocol, `SystemClock`, deterministic `LogicalClock`, `build_clock`. |
| `contracts.py` | Added `TrustSnapshot.blocked`, `VerificationResult.evidence_doc_ids`/`error`, and the optional `ChatLLM` protocol. |
| `security.py` | **New** `attach_security()` plus the `PROTECTED`/`DEGRADED`/`UNPROTECTED` health report. **This pass:** embedding probe checks shape, configured dimensionality, finite values and non-zero norm; generation probe performs a two-token completion (OpenAI-compatible `/chat/completions`, Ollama `/api/generate`, or in-process HF `generate`) after metadata checks; NLI probe runs one fixed inference and validates label/score; `research_ready` requires earned `PROTECTED`; custom verifiers report `unverified`, not false success. |
| `pipeline.py` | Secure by default; `secure=False` maps to an honest UNPROTECTED report. **This pass:** construction closes the store if any builder step fails; `close()` is idempotent and always closes the security stack, store and answer log. |
| `cli.py` | Added `doctor` (control/backend report) and `smoke` (matrix) subcommands; query warns when not protected. |
| `ingestion/pipeline.py` | Ingestor accepts the shared clock and uses it for default timestamps. |
| `retrieval/retriever.py` | Defence in depth: drops QUARANTINED/REJECTED documents even if the blocked cache is stale; reports stale-index ids and raises when every candidate is missing. |
| `generation/prompts.py` | Role-separated `build_answer_messages`/`build_single_doc_messages`, XML-delimited evidence, escaping of `</document`, explicit untrusted-data policy. |
| `generation/llm.py` | `generate_chat` for stub/OpenAI-compatible/Ollama/HF backends; flattened fallback via `generate_with_messages`; stub ignores question-only sentences when extracting claims. |
| `generation/grounded.py` | Drops blocked passages before generation; records excluded ids and timestamps; uses chat roles when available. |
| `trust/ledger.py` | Injected clock; SQLite busy timeout; evidence deduplication; **this pass:** `close()` is idempotent, failed construction closes the connection, and a finalizer is a last-resort cleanup. |
| `trust/policy.py` | Full state machine: blocked-status guard, fail-closed MEDIUM, deduplicated trust updates, inline/queued shared `_apply_outcome`, unified quarantine remediation. |
| `trust/queue.py` | Bounded FIFO dedupes by doc; records errors and malformed results as NEUTRAL with `error`; evidence-keyed ledger updates. |
| `trust/verifier.py` | Evidence recording, counterfactual influence, independent-source filtering, topical refutation gate, person-name conflict detection, trust-aware refutation, lexical/NLI aggregation. **This pass:** `nli_probe()` exercises the loaded cross-encoder and validates the outcome contract. |
| `ingestion/provenance.py` | **This pass:** idempotent close, failed-constructor cleanup, finalizer; no behaviour change to ingestion. |
| `provenance/answer_log.py` | **This pass:** idempotent close, failed-constructor cleanup, finalizer; no change to recorded answers. |
| `ingestion/subset.py` | **This pass:** bounded-memory exact subset construction, atomic output, duplicate/blank ID rejection, strict qrels/query validation, and exact size/unique ID/100% gold verification. |
| `snapshots.py` | **This pass:** staged backup/restore, signature-bound metadata, manifest hash/index file-set validation, checkpoint-last commit marker, recoverable `.previous` index and interrupted-staging cleanup. |

### Person C: attacks, baselines, evaluation

| File | Change |
|---|---|
| `attacks/poisoned_rag.py` | Deterministic entity-swap, negation and instruction-injection generators; `make_poisoned_document`; stable serialization. |
| `attacks/adaptive.py` | `slow_burn`, `hit_and_run`, `framing`, `signal_evasion`, attack event serialization and schedule helpers. |
| `attacks/stream.py` | Canonical query/ingestion rows, legacy adapters, `StreamContractError`, `zipf_queries`, schedule builders, ordering validation. |
| `baselines/policies.py` | Eight baselines on one policy interface; reads `t_eff`; shared `BaselineResult`. |
| `evaluation/chronological.py` | Stream execution over the real pipeline; eligibility is fixed before outcomes; exposure and citation use the same eligible-attempt denominator; answer-level ASR emitted as `None` + `NOT MEASURED`. |
| `evaluation/metrics.py` | **This pass (verified, not rewritten):** `poison_exposure_rate(exposed, eligible_attempts)`, `poison_citation_rate(cited, eligible_attempts)`, explicit `answer_level_asr_fields`, zero-denominator `None`, detection and timing metrics. |
| `evaluation/runner.py` | `EvaluationResult` serializes numerator/denominator for exposure, citation and answer-level ASR; defaults answer-level ASR to `NOT MEASURED`. |
| `evaluation/experiment.py` | **New** fixture loader, attacks, per-baseline pipelines, `execute_case`, matrix, CSV/markdown. CSV includes eligible/exposed/cited counts, not a bare "ASR". |
| `evaluation/integration.py` | Uses `attacks.stream` canonical rows. |
| `evaluation/__init__.py` | Exports the metrics/experiment surface. |

### Tests and documentation

| File | Change |
|---|---|
| `tests/test_security_wiring.py` | **Extended this pass:** 12 readiness cases — healthy embedding, NaN/zero/wrong-dim vectors, metadata-only generator, generation failure, valid completion, NLI inference failure, valid NLI outcome, requested-NLI-to-lexical fallback, unavailable ledger, absent queue, offline backends, full PROTECTED, custom verifier unverified, and security-disabled UNPROTECTED. Mocks simulate operations, not `ready=True` flags. |
| `tests/conftest.py` | **This pass:** autouse teardown closes any test-owned `AnswerLog`, `ProvenanceStore` or `TrustLedger`, even when an assertion fails; every close is idempotent. |
| `tests/test_chronological_smoke.py` | Stream adapter/contract, Zipf, full entity-swap lifecycle and matrix output tests. |
| `tests/test_person_c.py` | Attack registry, baselines, metrics, statistics and experiment executor coverage. |
| `tests/test_trust.py` | Evidence deduplication, logical clock, trust/policy/verifier coverage. |
| `tests/test_trust_plan_gaps.py` | Queue drain, background worker, admin review, trust propagation and plot tests. |
| `tests/test_verifier_nli.py` | NLI/lexical false-positive regressions. |
| `tests/test_generation.py` | Chat-role separation, document-block escaping, blocked-passage filtering, and generation backend failure recorded as an abstention. |
| `tests/test_resource_lifecycle.py` | **New this pass:** repeated close is a no-op for all three SQLite components; a constructor that fails after opening SQLite closes its connection; pipeline shutdown can be repeated. |
| `tests/test_snapshots.py` | **New this pass:** 16 tests for atomic backup/restore, corruption, interruption, signature mixing, index rollback and read-only summaries. |
| `tests/test_ingestion_subset.py` | **New this pass:** 34 direct Parquet tests for exact size, unique IDs, 100% gold coverage, determinism, seeds, all-gold, missing gold, insufficient rows, duplicate IDs, invalid queries and both qrels layouts. |
| `tests/test_research_preflight.py` | **Extended this pass:** real CLI doctor into disposable storage; fresh experiment directory remains empty; report records isolation and cleanup. |
| `tests/test_colab_workflow.py` | **Extended this pass:** notebook preflight isolation/opt-in assertions and a stale-pin guard (ancestor, pin ≠ HEAD, no implementation/test diff, sentinel code at pin). |
| `notebooks/trace_rag_colab.ipynb` | **This pass:** preflight uses `PREFLIGHT_PARENT` and asserts isolation; raw NQ download is opt-in and validates all three files; pin updated to the tested implementation commit. |
| `README.md` | Rewritten for the integrated system; **this pass:** 250/80 evidence, functional readiness explanation, exposure/citation table, NOT-MEASURED ASR, current limitation wording. |
| `docs/REAL_MODEL_EVAL.md` | **New** NQ/GPU runbook; **this pass:** explicit status box (subset, not full corpus, not executed, answer ASR not measured). |
| `docs/ENGINEERING_REPORT.md` | This report, regenerated from the current execution logs. |
| `"tall pytest pytest-cov"` | Deleted tracked junk file from the original commit. |

---

## C. Defect register

| # | Severity | Defect | Root cause | Fix | Regression test |
|---|---|---|---|---|---|
| 1 | Critical | Documented stream adapters emitted `{text, qid, t}` but `run_stream` required `{query, query_id, step}`. | Two independently invented schemas. | Canonical rows in `attacks/stream.py`; all conversions through adapters. | `test_stream_adapters_accept_legacy_and_canonical_rows` |
| 2 | Critical | `run_trust_stream.py` crashed on import: `zipf_queries` did not exist. | Function referenced before implementation/export. | Implemented deterministic Zipf sampler and exported it. | `test_zipf_queries_is_deterministic_and_returns_known_questions` |
| 3 | High | Trust/robust baselines read `doc.trust.trust`, which `TrustSnapshot` does not define. | Baselines written against an older object. | Baselines read `t_eff`. | `test_trust_and_robust_baselines_are_distinct` |
| 4 | Critical | `from_config()` built a null trust provider and `DefaultPolicy`. | Security was opt-in wiring. | `attach_security()` runs by default; `secure=False` is explicit. | `test_default_config_enables_the_security_stack` |
| 5 | High | Queued refutations updated trust but never triggered quarantine remediation. | Inline and queued paths duplicated logic. | Shared `_apply_outcome`; drain report carries newly quarantined ids. | `test_queued_refutation_triggers_the_same_remediation_callback` |
| 6 | High | MEDIUM + NEUTRAL was used; missing verifier failed open. | Policy conflated unresolved with clean. | Default withhold; fail-closed missing verifier. | `test_medium_band_without_verifier_fails_closed` |
| 7 | High | Lexical verifier treated an off-topic passage as a contradiction. | No topical floor. | `min_refute_topical_overlap` on both paths. | `test_off_topic_passage_cannot_refute` |
| 8 | High | A passage that omitted the designer's name was treated as a rival answer. | `name_conflict` tested absence, not a rival person. | Person-position detection; silence neutral. | `test_lexical_silence_is_not_a_rival_answer` |
| 9 | Medium | A less-trusted fresh source could refute a more-established target. | Refutation ignored relative source trust. | `_may_outrank` gate. | `test_less_trusted_source_cannot_refute_a_trusted_target` |
| 10 | High | A quarantined document could reach generation if the caller assembled context. | Policy was the only enforcement point. | `GroundedGenerator` filters blocked snapshots. | `test_generator_never_uses_a_quarantined_passage` |
| 11 | High | Passages and policy were flattened into one prompt string. | Prompt-only LLM interface. | Role-separated messages; backends implement `generate_chat`. | `test_chat_prompt_keeps_policy_and_evidence_in_separate_roles` |
| 12 | Medium | Duplicate verification evidence counted twice. | No idempotency key. | `trust_observations` unique `(doc_id, evidence_key)`. | `test_duplicate_evidence_key_is_idempotent` |
| 13 | Medium | Simulations read the wall clock. | `time.time()` scattered through ledger/ingestor. | Injected `Clock`. | `test_logical_clock_makes_snapshots_deterministic` |
| 14 | Medium | Zero-denominator rates returned fake `0.0`. | Direct division in metrics. | `_ratio` returns `None`; explicit rendering. | `test_metrics_and_zero_denominators` |
| 15 | Medium | Scripts opened a second ledger on the same file. | Re-wiring instead of reusing the factory stack. | Reuse `pipeline.security`; stream refuses unprotected. | Scripts exit 0 (`baseline_logs/`) |
| 16 | **High** | The smoke table reported **ASR 0.202**; the numerator was poison reaching the generation context, not a successful answer manipulation. | One metric name covered retrieval, citation and answer success; the harness inferred more than it measured. | `EvaluationResult` now carries `eligible_attack_attempts`, `poison_exposed_attempts/rate`, `poison_cited_attempts/rate` and `answer_level_asr=None` with `NOT MEASURED`. Results were regenerated; the old number is relabelled exposure. | `test_poison_rates_use_eligible_attempts_and_hand_calculated_counts`, `test_answer_level_asr_is_explicitly_not_measured`, matrix output asserts `attack_success_rate` absent and `answer_level_asr_status == "NOT MEASURED"` |
| 17 | **High** | `doctor` trusted model names, `/models` metadata and `nli_ready` booleans; a reachable endpoint with a broken completion or an unloaded NLI model could look research-ready. | Readiness checked configuration, not operations. | Functional probes: finite embedding, two-token completion, fixed NLI inference with label/score validation; `PROTECTED` requires all to pass; `research_ready` requires `PROTECTED`. **This pass:** HF empty/whitespace rejected, OpenAI/Ollama completion text validated, HTTP timeouts failed closed, credentials redacted, malformed/missing NLI outputs rejected, NLI-not-loaded is `failed`. | 20 cases in `tests/test_security_wiring.py` plus preflight tests |
| 18 | **Medium** | Repeated `ResourceWarning: unclosed database`; abandoned ledgers/stores/logs leaked SQLite connections. | Clearance depended on every test closing its own object; failed paths never did. | Idempotent `close()` on `TrustLedger`, `ProvenanceStore`, `AnswerLog`; failed constructors close partial connections; `pipeline.close()` closes all owned resources; finalizers are a last resort; test teardown closes leftovers. | Full suite run with `-W error::ResourceWarning` reports 0 warnings |
| 19 | **Critical** | Functional preflight created `answers.sqlite3`, `provenance.sqlite3` and `trust.sqlite3` in a fresh experiment directory; the runner then rejected its own clean run as "partial artifacts". | The doctor ran the real pipeline against the experiment's `storage.root`. | `--preflight-storage-root` plus `isolate_storage_config()`: unique disposable probe tree, model/security config unchanged, absolute/`..` storage paths rebased, isolation recorded, probe removed. | `test_functional_doctor_uses_isolated_storage_and_leaves_fresh_run_empty`, notebook assertion on `storage_isolation` |
| 20 | **High** | Subset construction could emit duplicate IDs, silently skip malformed qrels/query rows, and was tested only with a loose size range. | No identifier/row validation, in-place output, and weak assertions. | Duplicate/blank ID rejection, strict qrels/query validation, atomic write, exact-count/unique-ID/100%-gold verification, direct Parquet tests. | 34 tests in `tests/test_ingestion_subset.py`; `test_subset_always_keeps_gold_passages` now asserts exactly 300 |
| 21 | **High** | Snapshot restore replaced files in place and metadata was not tied to a run signature; an interrupted restore or another experiment's snapshot could be mixed in. | No staging/commit marker and no signature binding. | Staged restore, `checkpoint.json` installed last, `run_signature`/manifest hash/index digest validation, signature refusal, recoverable `.previous` index, interrupted-staging cleanup. | 16 tests in `tests/test_snapshots.py`, including corruption, interruption and signature mixing |
| 22 | **Medium** | Notebook pinned stale commit `2bac018…` and downloaded the multi-gigabyte NQ corpus implicitly. | Pin and data switch were not release-gated. | Pin updated to the tested implementation commit; `RUN_DATASET_DOWNLOAD = False`; raw files validated; stale-pin regression requires the pin to be an ancestor of HEAD, different from HEAD, with no implementation/test diff and sentinel code present at the pin. | `test_notebook_pin_contains_the_test_suite_implementation`, `test_notebook_isolates_preflight_storage_and_requires_data_opt_in` |

**Residual defect, not claimed fixed:** the lexical verifier still produces
false positives when a correct passage and a poison passage have equal source
trust and the only supporting copy is excluded as a near-duplicate. One clean
`newcomer` passage was quarantined in `hit_and_run` and `signal_evasion`
(clean FPR 0.026). This is recorded below, not hidden.

---

## D. Architecture confirmation

The code path actually executed by the supported entry points:

```
PersonAPipeline.from_config(config)
  ├─ ProvenanceStore / embedder / index / scorer / generator / AnswerLog
  └─ attach_security(pipeline)                     [security.py]
       ├─ TrustLedger(config.trust_db_path(), clock=...)
       ├─ CorroborationVerifier(llm=pipeline.generator.llm, use_nli=...)
       ├─ VerificationQueue(max_size=...)
       └─ TrustPolicy(ledger, queue, on_quarantine=pipeline.on_quarantine)

pipeline.answer(query, query_id, step)
  ├─ TrustWeightedRetriever.retrieve
  ├─ SignalComputer.compute + scorer.score_snapshot/band
  ├─ TrustPolicy.decide
  │    ├─ LOW → context
  │    ├─ MEDIUM → CorroborationVerifier.verify
  │    │    ├─ _extract_claim / _check_influence / _counterfactual_influence
  │    │    ├─ _find_independent (source, family, burst rules)
  │    │    └─ _corroborate_lexical / _corroborate_nli → SUPPORT/REFUTE/NEUTRAL
  │    ├─ _apply_outcome → TrustLedger.record_observation
  │    ├─ HIGH → record_high_band + queue.enqueue
  │    └─ _remediate → pipeline.on_quarantine → RemediationService
  ├─ GroundedGenerator.generate
  ├─ AnswerLog.record
  └─ drain_verification_queue (after the answer)

evaluation.run_stream / run_chronological_eval
  ├─ Ingestor.ingest_text → ProvenanceStore
  ├─ pipeline.index_chunks → index
  ├─ pipeline.answer for each StreamQuery
  └─ metrics + per-step JSONL/JSON + summary
```

### Readiness probes and status classification

`security.py` now distinguishes three things that were previously conflated:

| Concept | What it means | Where it appears |
|---|---|---|
| **Configuration** | The requested backends, paths, verifier mode and thresholds | `configuration_valid` |
| **Component readiness** | Each already-constructed component passed a functional probe | `readiness_probes`, `readiness_ready` |
| **Research readiness** | `status == PROTECTED` and every required probe passed | `research_ready` |

Functional probes, all bounded and side-effect free apart from one tiny
inference call:

* **Embedding:** encode `"readiness probe"`; require shape `(1, dim)`, the
  configured dimensionality, only finite values and a non-zero norm. `hashing`
  is allowed to pass the operation but is reported as **fallback**.
* **Generation:** after checking `/models` or `/api/tags` metadata, request a
  two-token completion with a fixed harmless prompt. OpenAI-compatible and
  Ollama responses must contain non-empty text; HF must have loaded
  tokenizer/model and return a response. The served model and response length
  are recorded. `stub` is **fallback**, never ready.
* **NLI:** `CorroborationVerifier.nli_probe()` runs one fixed inference on the
  already-loaded cross-encoder and validates that the label maps to
  entailment/contradiction/neutral and the score is a finite probability. A
  loaded model that crashes is **failed**, not ready. No probe downloads a
  model.
* **Trust ledger:** must be a `TrustLedger` with `SELECT 1` succeeding and a
  persistent path; in-memory is fallback.
* **Queue:** must exist, have positive capacity and be readable.

Status rules:

* `UNPROTECTED` — security disabled, or the ledger/policy/verifier/queue is not
  attached, or the persistent ledger/queue probe fails.
* `DEGRADED` — controls are attached, but a required backend is fallback,
  unavailable or not functionally verified (including lexical instead of NLI).
* `PROTECTED` — every required probe passed. This is the only state in which
  `research_ready` is true.

On this offline checkout `doctor` reports `DEGRADED` with embedding, generation
and NLI fallbacks, and ledger/queue ready; `readiness_failures` is empty because
the offline backends are documented fallbacks, not crashes.

Key contracts:

* `RetrievedDocument.trust` is a `TrustSnapshot`; retrieval and baselines read
  `t_eff`.
* Stream rows are canonical `{step, query_id, query}` and
  `{step, doc_id, text, source_id, ...}`; adapters accept legacy rows.
* `VerificationResult` records evidence ids and errors, so a failed backend
  cannot grant trust.
* A `QUARANTINED`/`REJECTED` snapshot is enforced at retrieval, policy and
  generation.
* Exposure and citation denominators are fixed to **eligible attack attempts**
  (target queries with targeted poison ingested before the query), including
  abstentions, generation failures and poison that was never retrieved.

---

## E. Test report

### Commands and environment

```bash
cd RAGLedger
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"

pytest -o addopts="" -q
pytest -o addopts="" -q --cov=trace_rag --cov-report=term-missing
python -m compileall -q src scripts tests
python scripts/check_setup.py
```

Environment: Python 3.13.3, Windows 11, numpy 2.5.4, scikit-learn 1.9.1,
pydantic 2.14.1, PyYAML 6.0.3, pytest 9.1.1, pytest-cov 7.1.0, pyarrow 26.0.0.
No GPU. `faiss-cpu`, `torch`, `transformers`, `matplotlib` absent.

### Results

| Check | Result |
|---|---|
| `pytest -o addopts="" -q` | **340 passed, 6 skipped** |
| Coverage | **81%** total (6299 statements, 1212 missed) |
| `-W error::ResourceWarning` | **0 warnings**; no unclosed SQLite connections |
| `compileall` | exit 0 |
| `check_setup.py` | exit 0: pyarrow found; faiss/torch/transformers/Ollama/vLLM warnings expected on this host |
| Notebook JSON/AST/pin | 30 cells, 15 code cells parse, no saved outputs; pin is an ancestor of HEAD |
| Offline mini stream + `--backup-dir` | exit 0; 4/4 steps, 3 distinct queries; snapshot chain and manifest written |

Skipped tests:

| Test | Reason |
|---|---|
| `test_index_and_retrieval.py` (1) | `faiss-cpu` not installed |
| `test_regressions.py` (1) | `faiss-cpu` not installed |
| `test_model_backends.py` (1) | `torch` not installed |
| `test_trust_plan_gaps.py` (3) | `matplotlib` not installed |

Coverage caveats:

* `index/faiss_index.py` 0% (library absent), `embeddings/hf_embedder.py` 0%
  (torch absent), `generation/llm.py` 56% (HTTP/HF paths not exercised),
  `trust/plots.py` 28% (matplotlib absent), `security.py` 88% (network
  branches and finalizers are not all exercised offline).
* New code coverage: `ingestion/subset.py` 91%, `snapshots.py` 86%,
  `provenance/remediation.py` 100%.
* These are environment gaps, not passes. They are listed in limitations.

This pass's validation logs were captured outside the repository for the exact
tested commit (pytest, coverage, ResourceWarning, compileall, notebook check and
mini stream). They are reproducible with the commands above; no Colab log is
claimed. The mini-stream manifest records the tested Git commit and
`run_kind: offline_smoke`.

---

## F. Evaluation report

### Scope

**SMOKE TEST.** Bundled synthetic mini corpus (18 clean passages + 10 late
clean + poison constructed per case), hashing embedder (256-d), numpy index,
stub LLM, lexical verifier, heuristic scorer, production trust thresholds
(`quarantine_refutations=2`, `reject_refutations=3`). No real model, no NQ, no
PoisonedRAG benchmark corpus.

Command:

```bash
python scripts/run_chronological_eval.py --out runs/final_smoke
```

Configuration: 7 attacks × 8 baselines × 1 seed (`20260921`), `target_limit=6`,
`repeat_targets=2`, 56 cases. Result: **56/56 completed, 0 failures**.
Every case reports `answer_level_asr: null`, `answer_level_asr_status:
"NOT MEASURED"`.

### RAGLedger by attack

| Attack | Eligible | Exposed (rate) | Cited (rate) | Precision | Recall | Clean FPR | Detection delay | LLM calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entity_swap | 12 | 0 (0.000) | 0 (0.000) | 1.000 | 0.333 | 0.000 | 12 | 123 |
| negation | 12 | 4 (0.333) | 4 (0.333) | 0.000 | 0.000 | 0.000 | — | 123 |
| instruction_injection | 12 | 5 (0.417) | 3 (0.250) | 0.000 | 0.000 | 0.000 | — | 129 |
| slow_burn | 12 | 0 (0.000) | 0 (0.000) | 0.500 | 0.167 | 0.020 | 19 | 131 |
| hit_and_run | 12 | 2 (0.167) | 2 (0.167) | 0.500 | 0.167 | 0.026 | 19 | 115 |
| signal_evasion | 12 | 2 (0.167) | 2 (0.167) | 0.500 | 0.167 | 0.026 | 19 | 137 |
| framing | 12 | 4 (0.333) | 4 (0.333) | 0.000 | 0.000 | 0.000 | — | 123 |

Mean over attacks: poison exposure **0.202**, poison citation **0.179**, clean
FPR **0.010**, recall **0.119**, LLM calls **125.9**. The old label "ASR 0.202"
was the exposure rate; it is not answer success. Aggregated over the seven
cases, document-level detection has 5 true positives and 3 false positives
(precision 0.625). All three false positives are `late_clean_3#0000`: it is
quarantined in `hit_and_run` and `signal_evasion`, and flagged but not
quarantined in `slow_burn`. Detection delay is the mean over detected documents
only; `—` means no detection.

### Baseline comparison (mean over the seven attacks)

| Baseline | Mean exposure | Mean citation | Mean clean FPR | Mean recall | Mean LLM calls |
|---|---:|---:|---:|---:|---:|
| no_defence | 0.929 | 0.643 | 0.000 | 0.000 | 29.0 |
| duplicate_filter | 0.929 | 0.643 | 0.000 | 0.000 | 29.0 |
| trust_threshold | 0.929 | 0.643 | 0.000 | 0.000 | 29.0 |
| perplexity | 0.929 | 0.643 | 0.000 | 0.000 | 29.0 |
| TrustRAG | 0.857 | 0.643 | 0.000 | 0.000 | 29.0 |
| RobustRAG | 0.929 | 0.643 | 0.000 | 0.000 | 29.0 |
| always_on_loo | 0.762 | 0.500 | 0.052 | 0.238 | 296.4 |
| **RAGLedger** | **0.202** | **0.179** | **0.010** | **0.119** | **125.9** |

These baselines are implemented against the same policy interface and run on the
same fixture/schedule; they are not the published systems' full pipelines, and
the toy corpus makes the comparison illustrative only. They measure exposure and
citation, not answer-level attack success.

### Trust-stream smoke (real pipeline, mini corpus)

Command:

```bash
python scripts/run_trust_stream.py --config config/default.yaml \
  --corpus examples/mini_corpus.jsonl --queries examples/mini_questions.json \
  --steps 12 --targets 2 --out runs/final_trust_stream
```

Observed (`runs/final_trust_stream/metrics.json`):

| Metric | Value |
|---|---:|
| eligible attack attempts | 6 |
| poison retrieved rate | 1.000 |
| poison exposure rate | 0.000 |
| poison citation rate | 0.000 |
| poison quarantined | 0 |
| clean false quarantine | 0 |
| verifier outcomes | SUPPORT 0, REFUTE 0, NEUTRAL 22 |
| mean LLM calls/query | 4.667 |
| mean latency | 17.57 ms |
| answer-level ASR | `null` (`NOT MEASURED`) |

The poison was retrieved but withheld from the generation context and never
cited. It was also **not detected or quarantined**: with the lexical fallback the
only contradicting copy was a near-duplicate in the same family, which the
independence rule correctly excludes. This is a limitation of the lexical
configuration on this fixture, not evidence that the defence works.

### Demos

| Command | Result |
|---|---|
| `python scripts/demo_end_to_end.py --root runs/demo` | exit 0; unprotected Person A demo |
| `python scripts/demo_person_b.py --root runs/final_demo_b` | exit 0; 3 poison docs quarantined; queue drained; later context contains clean documents only |
| `trace-rag doctor` | exit 0; `DEGRADED`, lexical verifier, queue on, trust persistent, no probe failures |
| `python scripts/run_chronological_eval.py --out runs/final_smoke` | exit 0; 56/56 cases |

Demo B uses a **scripted** verifier for stage 5 (documented in its output) to
exercise queue → quarantine → remediation without a model. It is a lifecycle
demonstration, not a detection result.

### What was not measured

No Contriever, LLM, DeBERTa NLI, FAISS, full NQ corpus or multi-seed run. No
real answer-level ASR, detection FPR, detection delay, remediation recall,
latency or cost numbers. The harness has no validated attack-specific
malicious-answer criterion, so answer-level ASR remains `NOT MEASURED`. See
`docs/REAL_MODEL_EVAL.md`.

---

## G. Limitations

1. **No real-model or full-dataset result.** The only evaluation is the
   synthetic smoke matrix and the mini trust stream. Do not report them as NQ,
   PoisonedRAG or research results.
2. **Answer-level ASR is not measured.** No attacker-target criterion is
   validated for these attacks, so no answer-success number exists. Exposure and
   citation must not be called ASR.
3. **Lexical verifier false positives.** In `hit_and_run` and `signal_evasion`,
   the correct `late_clean_3#0000` passage was quarantined after two refutations
   from an equal-trust attacker source. The clean FPR is 0.026 in those cases.
   The trust-aware gate prevents a strictly less-trusted witness, but equal
   trust remains ambiguous.
4. **High poison exposure for negation, instruction injection and framing**
   (0.33–0.42). The cheap signals do not reliably flag these transformations,
   and the lexical verifier often returns NEUTRAL. `instruction_injection` also
   depends on the generator's instruction-following; the stub LLM is not a
   realistic target.
5. **The mini corpus is trivially separable.** All signals can separate the toy
   poison; the scorer demo reports perfect held-out scores that mean nothing.
6. **Optional backends unexercised.** FAISS, HuggingFace embeddings, torch,
   transformers and matplotlib are not installed. Their tests are skipped.
   `pyarrow 26.0.0` was installed only for the Parquet subset tests; the Colab
   target is `pyarrow==17.0.0`, which remains unverified. `run_trust_stream`
   figures are skipped without matplotlib. The functional readiness probes for
   real backends are unit-tested with simulated operations but have not been run
   against real models here.
7. **No real-NQ attack/baseline matrix.** `run_chronological_eval.py` is
   fixture-only. `run_trust_stream.py` is the available real-data end-to-end
   path and does not compute answer-level ASR.
8. **Suspicion scorer training on real data.** The real runner uses the
   heuristic scorer unless a trained `scorer.joblib` and labelled rows are
   supplied. There is no real-NQ labelled-row generator in this repository.
9. **Single seed.** The smoke matrix uses one seed; no confidence intervals or
   paired significance tests are reported for it. The statistics module exists
   and is unit-tested, but not applied to a multi-seed real experiment.
10. **Historical docs.** `docs/INTERFACES.md`, `docs/HANDOVER*.md` and
    `docs/VERIFICATION.md` retain Person A-era statements and counts. Code,
    README and this report are authoritative.
11. **Data/licence.** No NQ corpus or model weights are committed. Check the
    dataset and model licences before redistribution.

---

## H. Reproduction instructions

From a clean checkout:

```bash
cd RAGLedger

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e ".[dev]"

pytest -o addopts="" -q
pytest -o addopts="" -q --cov=trace_rag --cov-report=term-missing
pytest -o addopts="" -q -W error::ResourceWarning
python -m compileall -q src scripts tests

python scripts/check_setup.py          # exit 0; optional backends warn
trace-rag doctor                       # expect DEGRADED in this offline setup

python scripts/run_chronological_eval.py --out runs/final_smoke
python scripts/demo_end_to_end.py --root runs/demo
python scripts/demo_person_b.py --root runs/final_demo_b

python scripts/run_trust_stream.py --config config/default.yaml \
  --corpus examples/mini_corpus.jsonl --queries examples/mini_questions.json \
  --steps 12 --targets 2 --out runs/final_trust_stream \
  --backup-dir runs/final_trust_stream_backups
```

Expected: **340 passed, 6 skipped**; the pin test passes only after the
notebook commit that follows the tested implementation commit; all scripts exit
0; `doctor` reports `DEGRADED` with no readiness failures; the ResourceWarning
run is clean. `runs/final_smoke/summary.md` carries the explicit smoke-scope
statement and `answer_level_asr_status = NOT MEASURED`.

For the real-model run, follow `docs/REAL_MODEL_EVAL.md`. A reportable run
requires `doctor` to say `PROTECTED` after functional probes (real embedding,
generation completion, NLI inference), `verifier_mode=nli`,
`trust_persistent=true`, `queue_enabled=true`, and a persistent ledger. Do not
label a `DEGRADED` or `UNPROTECTED` run as RAGLedger.

### Captured evidence

This pass's logs were written outside the repository so they cannot drift from
the tested tree. They record the exact `git rev-parse HEAD` in the mini-stream
manifest and were captured for commit
`58de432c3124cc7cf81dcbfc4cc169b91c00af08`:

```
pytest.txt
coverage.txt
resource_warnings.txt          # -W error::ResourceWarning, 0 warnings
compileall.txt
notebook_check.txt             # JSON/AST/pin, no saved outputs
mini_stream.txt
mini_stream/active/run_manifest.json
check_setup.txt
```

Historical committed evidence remains in `baseline_logs/` and describes older
commits; it is not evidence for this pass. No Colab or real-model log exists.
They are not portable measurements of another machine or GPU.
