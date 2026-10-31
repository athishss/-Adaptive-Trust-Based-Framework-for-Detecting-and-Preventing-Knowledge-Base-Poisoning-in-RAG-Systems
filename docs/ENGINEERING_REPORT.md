# RAGLedger engineering report

**Repository:** `RAGLedger` (TRACE-RAG adaptive trust framework)
**Audit date:** 2026-10-23
**Environment:** Windows 10, CPython 3.13.3, no GPU; core dependencies only
(numpy 2.5.4, scikit-learn 1.9.1, pydantic 2.14.1, PyYAML). Optional backends
`faiss-cpu`, `torch`, `transformers`, `pyarrow` and `matplotlib` are **not
installed**.
**Scope:** integration audit, repairs, deterministic offline verification, and a
runbook for the real-model experiment. No real-model run was performed.

---

## A. Executive summary

The repository began as three partially integrated workstreams. Person A's
pipeline was functional and its tests passed, but the documented end-to-end
entry points did not all execute:

* `scripts/run_trust_stream.py` failed on import because it imported
  `zipf_queries` from a module that did not define it;
* `evaluation/integration.py` emitted `{text, qid, t}` while the chronological
  runner read `{query, query_id, step}`, so the documented stream entry point
  failed with `KeyError: 'step'`;
* the trust/robust baselines read `doc.trust.trust`, a field that does not
  exist on `TrustSnapshot` (the contract exposes `t_eff`);
* `PersonAPipeline.from_config()` did not attach the trust ledger, verifier or
  policy, so the normal CLI/library path was an unprotected RAG system;
* Person C's attacks, baselines and evaluation had no executable end-to-end
  path (test coverage of those modules was 0%), and the default demo could not
  demonstrate queueing, quarantine and remediation together.

After the repairs:

| Check | Result |
|---|---|
| Full test suite | **235 passed, 6 skipped** |
| Coverage | **80%** overall |
| Python compilation | `compileall` exit 0 |
| Chronological matrix | **56/56 cases**, 0 failures |
| `demo_end_to_end.py` | exit 0 |
| `demo_person_b.py` | exit 0, 3 poison passages quarantined, later retrieval clean |
| `run_trust_stream.py` (mini, lexical) | exit 0, no false quarantine, **no poison quarantine** (honest limitation) |
| `trace-rag doctor` | `DEGRADED` (controls active; offline models) |
| Real-model NQ run | **not run** |

The system is now integrated and reproducible at smoke scale. It is **not** a
validated benchmark result: the lexical verifier misses several attack families,
one clean passage was falsely quarantined in two smoke cases, and no real-model
or full-dataset experiment has been executed.

---

## B. File-by-file change log

### Configuration and scripts

| File | Change |
|---|---|
| `config/default.yaml` | Added `time`, `security` and `trust` sections; security enabled by default with lexical verifier, fail-closed MEDIUM handling, queue and evidence thresholds. |
| `config/nq_gpu.yaml` | Added the same security/trust surface with `verifier: nli`, persistent ledger, queue and production quarantine thresholds. |
| `config/full_nq.yaml` | Same real-model security configuration for the full-corpus IVF-PQ setup. |
| `scripts/demo_end_to_end.py` | Now explicitly builds an **UNPROTECTED** Person A demo (`secure=False`) and prints that mode, so it cannot be mistaken for the protected system. |
| `scripts/demo_person_b.py` | Reuses the security stack attached by `from_config` instead of opening a second ledger; adds a clearly labelled scripted verifier solely to exercise queue → refute → quarantine → remediation. |
| `scripts/run_trust_stream.py` | Imports `zipf_queries` from its real home; configures trust via `config` rather than replacing the attached ledger; refuses to run if the stack did not attach; makes figures optional so a missing matplotlib cannot lose a completed run. |
| `scripts/run_chronological_eval.py` | **New** supported matrix entry point (7 attacks × 8 baselines, deterministic fixture, machine-readable outputs). |

### Core integration and time model

| File | Change |
|---|---|
| `src/trace_rag/config.py` | Added `TimeConfig`, `TrustSettings` and `SecurityConfig`; added `Config.trust_db_path()`; validation for quarantine/reject and threshold ordering. |
| `src/trace_rag/utils/clock.py` | **New** `Clock` protocol, `SystemClock`, deterministic `LogicalClock`, `build_clock`. All simulation timestamps now come from one place. |
| `src/trace_rag/contracts.py` | Added `TrustSnapshot.blocked`, `VerificationResult.evidence_doc_ids`/`error`, and the optional `ChatLLM` protocol. |
| `src/trace_rag/security.py` | **New** `attach_security()` plus the `PROTECTED`/`DEGRADED`/`UNPROTECTED` health report; installs ledger, verifier, policy and queue. |
| `src/trace_rag/pipeline.py` | Secure by default; maps `secure=False` to an honest UNPROTECTED health report; passes explicit timestamps; drains the queue after each answer and merges newly quarantined ids. |
| `src/trace_rag/cli.py` | Added `doctor` (control/backend report) and `smoke` (matrix) subcommands; query warns when the run is not protected. |
| `src/trace_rag/ingestion/pipeline.py` | Ingestor accepts the shared clock and uses it for default timestamps. |
| `src/trace_rag/retrieval/retriever.py` | Defence in depth: drops documents whose snapshot is QUARANTINED/REJECTED even if the provider's blocked list is stale; reports stale-index ids and raises when every candidate is missing. |
| `src/trace_rag/generation/prompts.py` | Role-separated `build_answer_messages`/`build_single_doc_messages`, XML-delimited evidence, escaping of `</document`, explicit untrusted-data policy. |
| `src/trace_rag/generation/llm.py` | `generate_chat` for stub/OpenAI-compatible/Ollama/HF backends; flattened fallback via `generate_with_messages`; stub ignores question-only sentences when extracting claims. |
| `src/trace_rag/generation/grounded.py` | Drops blocked passages before generation; records excluded ids and timestamps; uses chat roles when available. |
| `src/trace_rag/trust/ledger.py` | Injected clock; SQLite busy timeout; `trust_observations` evidence deduplication; `record_observation` returns whether the evidence was new. |
| `src/trace_rag/trust/policy.py` | Full state machine: blocked-status guard, fail-closed MEDIUM, deduplicated trust updates, inline/queued shared `_apply_outcome`, unified quarantine remediation, per-document decision reasons. |
| `src/trace_rag/trust/queue.py` | Bounded FIFO dedupes by doc; records errors and malformed results as NEUTRAL with `error`; evidence-keyed ledger updates. |
| `src/trace_rag/trust/verifier.py` | Evidence recording, error handling, counterfactual influence, independent-source filtering, topical refutation gate, person-name conflict detection, trust-aware refutation, lexical/NLI aggregation. |

### Person C: attacks, baselines, evaluation

| File | Change |
|---|---|
| `src/trace_rag/attacks/poisoned_rag.py` | Deterministic entity-swap, negation and instruction-injection generators; `make_poisoned_document`; stable serialization. |
| `src/trace_rag/attacks/adaptive.py` | `slow_burn`, `hit_and_run`, `framing`, `signal_evasion`, attack event serialization and schedule helpers. |
| `src/trace_rag/attacks/stream.py` | Canonical query/ingestion rows, legacy adapters, `StreamContractError`, `zipf_queries`, query/ingestion schedule builders, ordering validation. |
| `src/trace_rag/attacks/__init__.py` | Exports the full attack registry and stream API. |
| `src/trace_rag/baselines/policies.py` | Eight baselines on one policy interface: no defence, duplicate filter, trust threshold, n-gram perplexity, TrustRAG, RobustRAG, always-on LOO; reads `t_eff`; shared `BaselineResult`. |
| `src/trace_rag/baselines/__init__.py` | Exports the baseline registry and builders. |
| `src/trace_rag/evaluation/chronological.py` | Stream execution over the real pipeline, per-step records, band/verification/detection/remediation metrics, JSONL/JSON outputs. |
| `src/trace_rag/evaluation/experiment.py` | **New** fixture loader, target selection, attack schedule construction, per-baseline pipeline assembly, `execute_case`, `run_matrix`, CSV/markdown writers. |
| `src/trace_rag/evaluation/integration.py` | Replaced the incompatible `{text, qid, t}` conversion with adapters over `attacks.stream` canonical rows. |
| `src/trace_rag/evaluation/metrics.py` | Document-level detection metrics, zero-denominator-safe rates, latency percentiles, exposure/detection delays, cost/utility metrics. |
| `src/trace_rag/evaluation/runner.py` | `EvaluationResult.details` and serialization for case artefacts. |
| `src/trace_rag/evaluation/__init__.py` | Exports the new metrics/experiment surface. |

### Tests and documentation

| File | Change |
|---|---|
| `tests/test_chronological_smoke.py` | **New** stream adapter/contract test, Zipf test, full entity-swap lifecycle test, matrix output test. |
| `tests/test_person_c.py` | Attack registry, baselines, metrics, statistics and experiment executor coverage. |
| `tests/test_trust.py` | Added evidence-deduplication and logical-clock tests; existing trust/policy/verifier coverage. |
| `tests/test_trust_plan_gaps.py` | Queue drain, background worker, admin review, trust propagation and plot tests. |
| `tests/test_verifier_nli.py` | NLI/lexical false-positive regressions: off-topic, silence-as-conflict, less-trusted witness, equal-trust contradiction, entity swap. |
| `tests/test_security_wiring.py` | **New** secure-by-default wiring, UNPROTECTED baseline report, fail-closed MEDIUM test. |
| `tests/test_generation.py` | Added chat-role separation, document-block escaping, and blocked-passage filtering tests. |
| `tests/conftest.py` | Clarified that the Person A fixture is intentionally UNPROTECTED; the secure default is covered separately. |
| `README.md` | Rewritten for the integrated system: setup, architecture, config, verified smoke results, limitations, real-model links. |
| `docs/REAL_MODEL_EVAL.md` | **New** exact NQ/GPU runbook and acceptance criteria. |
| `docs/ENGINEERING_REPORT.md` | This report. |
| `"tall pytest pytest-cov"` | Deleted tracked junk file (a captured terminal log with no code value) from the original commit. |

---

## C. Defect register

| # | Severity | Defect | Root cause | Fix | Regression test |
|---|---|---|---|---|---|
| 1 | Critical | Documented stream adapters emitted `{text, qid, t}` but `run_stream` required `{query, query_id, step}`; entry point failed with `KeyError: 'step'`. | Two independently invented schemas with no shared contract. | Canonical rows in `attacks/stream.py`; all conversions go through `as_query_row`/`as_ingestion_row`; required-field validation. | `test_stream_adapters_accept_legacy_and_canonical_rows` |
| 2 | Critical | `run_trust_stream.py` crashed on import: `zipf_queries` did not exist in `attacks/stream.py`. | Person C function referenced before implementation/export. | Implemented deterministic Zipf sampler and exported it. | `test_zipf_queries_is_deterministic_and_returns_known_questions` |
| 3 | High | `trust_threshold_filter` and `RobustRAGPolicy` read `doc.trust.trust`, which `TrustSnapshot` does not define. | Baselines written against an older trust object. | Baselines read the contract field `t_eff`; registry test asserts the baselines are distinct. | `test_trust_and_robust_baselines_are_distinct` |
| 4 | Critical | `PersonAPipeline.from_config()` built a `NullTrustProvider` and `DefaultPolicy`; the CLI was unprotected despite shipping a security module. | Security was opt-in wiring, not part of the factory. | `attach_security()` runs by default; `secure=False` is the explicit baseline; health report and CLI warn. | `test_default_config_enables_the_security_stack`, `test_explicit_unprotected_baseline_is_reported_as_unprotected` |
| 5 | High | Queued (HIGH-band) refutations updated trust but never triggered the quarantine callback, so earlier answers were not remediated. | Inline and queued paths duplicated the state-machine logic and diverged. | Both call shared `_apply_outcome`; `QueueDrainReport` carries newly quarantined ids; policy invokes remediation once. | `test_high_band_is_queued_and_drained_off_path`, `test_queued_refutation_triggers_the_same_remediation_callback` |
| 6 | High | MEDIUM + NEUTRAL was treated as "benefit of the doubt" and used in the answer; a missing verifier also failed open. | Policy conflated "unresolved" with "clean". | Default `medium_neutral_action=exclude`; missing verifier fails closed unless the research flag is set. | `test_medium_band_without_verifier_fails_closed`, `test_medium_band_can_fail_open_only_when_configured` |
| 7 | High | Lexical verifier counted an off-topic passage as a full-confidence contradiction. | No topical floor before refutation. | `min_refute_topical_overlap` gate in both lexical and NLI paths. | `test_off_topic_passage_cannot_refute`, `test_lexical_mode_also_refuses_to_refute_off_topic` |
| 8 | High | A passage that merely omitted the designer's name was treated as a rival named answer, refuting and eventually quarantining correct content. | `name_conflict` tested absence of the target name instead of presence of a rival person. | Person-position detection including agent phrases (`by/of`); silence is neutral. | `test_lexical_silence_is_not_a_rival_answer` |
| 9 | Medium | A less-trusted fresh source could refute a more-established target source. | Refutation ignored relative source trust. | `_may_outrank` gate: equal trust may refute; strictly lower trust may not. | `test_less_trusted_source_cannot_refute_a_trusted_target`, `test_equal_trust_source_can_still_refute` |
| 10 | High | A quarantined document could reach the generator if a caller assembled the context list itself. | Policy was the only enforcement point. | `GroundedGenerator` filters blocked snapshots and records them as excluded. | `test_generator_never_uses_a_quarantined_passage` |
| 11 | High | Retrieved passages and system policy were flattened into one prompt string, so in-context instructions could act as policy. | Prompt-only LLM interface. | `build_answer_messages` keeps policy in `system` and evidence in `user`; backends implement `generate_chat`; flattened fallback retains delimiters and warnings. | `test_chat_prompt_keeps_policy_and_evidence_in_separate_roles`, `test_evidence_cannot_close_its_own_document_block` |
| 12 | Medium | The ledger counted the same verification evidence twice on retries/duplicate queue entries. | No idempotency key on observations. | `trust_observations` unique `(doc_id, evidence_key)`; `record_observation` returns False for duplicates. | `test_duplicate_evidence_key_is_idempotent`, `test_new_evidence_key_counts_as_a_new_observation` |
| 13 | Medium | Simulations read the wall clock in trust decay, bursts and source age, so runs were not reproducible. | `time.time()` calls scattered through ledger/ingestor. | Injected `Clock`; logical mode maps step → `epoch + step*seconds`. | `test_logical_clock_makes_snapshots_deterministic` |
| 14 | Medium | Zero-denominator rates returned fake `0.0`, hiding "not measured" as "perfect". | Direct division in metrics. | `_ratio` returns `None`; summary renders `None` explicitly. | `test_metrics_and_zero_denominators` |
| 15 | Medium | `demo_person_b.py` and `run_trust_stream.py` opened a second ledger on the same SQLite file after `from_config` had already attached one, producing `database is locked`. | Scripts re-wired components instead of reusing the factory stack. | Both scripts reuse `pipeline.security`/`pipeline.trust_provider`; `run_trust_stream` refuses to run unprotected. | Verified by both scripts exiting 0 (`baseline_logs/`) |

**Residual defect, not claimed fixed:** the lexical verifier still produces
false positives when a correct passage and a poison passage have equal source
trust and the only supporting copy is excluded as a near-duplicate. One clean
`newcomer` passage was quarantined in `hit_and_run` and `signal_evasion`
(clean FPR 0.026). This is recorded in the evaluation below and in the
limitations, not hidden.

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
  │    └─ score = similarity * max(t_eff, floor)^lambda; blocked docs removed
  ├─ SignalComputer.compute + scorer.score_snapshot/band
  ├─ TrustPolicy.decide
  │    ├─ LOW → context
  │    ├─ MEDIUM → CorroborationVerifier.verify
  │    │    ├─ _extract_claim
  │    │    ├─ _check_influence / _counterfactual_influence
  │    │    ├─ _find_independent (different source, family, not same burst)
  │    │    ├─ _corroborate_lexical / _corroborate_nli
  │    │    └─ _decide → SUPPORT / REFUTE / NEUTRAL
  │    ├─ _apply_outcome → TrustLedger.record_observation
  │    │    └─ _check_demotion → QUARANTINED
  │    ├─ HIGH → record_high_band + queue.enqueue
  │    └─ _remediate → pipeline.on_quarantine → RemediationService
  ├─ GroundedGenerator.generate (role-separated prompt, citations, abstention)
  ├─ AnswerLog.record
  └─ drain_verification_queue (after the answer)
       └─ queue.drain → verifier.verify → ledger.record_observation
            └─ newly quarantined → _remediate

evaluation.run_stream / run_chronological_eval
  ├─ Ingestor.ingest_text → ProvenanceStore
  ├─ pipeline.index_chunks → index
  ├─ pipeline.answer for each StreamQuery
  └─ metrics + per-step JSONL/JSON + summary
```

Key contracts:

* `RetrievedDocument.trust` is a `TrustSnapshot`; retrieval and baselines read
  `t_eff`, never an ad-hoc attribute.
* Stream rows are canonical `{step, query_id, query}` and
  `{step, doc_id, text, source_id, ...}`; adapters accept legacy rows.
* `VerificationResult` records `evidence_doc_ids` and `error`, so an audit can
  see which independent passages produced a verdict and a failed backend cannot
  grant trust.
* A `QUARANTINED`/`REJECTED` snapshot is enforced at retrieval, policy and
  generation (defence in depth).

---

## E. Test report

### Commands and environment

```bash
cd RAGLedger
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"

pytest -o addopts="" -q
pytest -o addopts="" -q --cov=trace_rag --cov-report=term
python -m compileall -q src scripts tests
python scripts/check_setup.py
```

Environment: Python 3.13.3, numpy 2.5.4, scikit-learn 1.9.1, pydantic 2.14.1.
No GPU. `faiss-cpu`, `torch`, `transformers`, `pyarrow`, `matplotlib` absent.

### Results

| Check | Result |
|---|---|
| `pytest -o addopts="" -q` | **235 passed, 6 skipped** in 13.94 s |
| Coverage | **80%** total |
| `compileall` | exit 0 |
| `check_setup.py` | exit 1: only `pyarrow` missing; torch/transformers/ollama/vLLM warnings are expected on this host |

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
  `trust/plots.py` 28% (matplotlib absent).
* These are environment gaps, not passes. They are listed in limitations.

Logs: `baseline_logs/pytest_final.txt`, `pytest_cov_final.txt`,
`compileall.txt`, `check_setup.txt`.

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
python scripts/run_chronological_eval.py --out runs/smoke_matrix
```

Configuration: 7 attacks × 8 baselines × 1 seed (`20260921`), `target_limit=6`,
`repeat_targets=2`, 56 cases. Result: **56/56 completed, 0 failures**.

### RAGLedger by attack

| Attack | ASR | Precision | Recall | Clean FPR | Detection delay | LLM calls |
|---|---:|---:|---:|---:|---:|---:|
| entity_swap | 0.000 | 1.000 | 0.333 | 0.000 | 12 | 123 |
| negation | 0.333 | 0.000 | 0.000 | 0.000 | — | 123 |
| instruction_injection | 0.417 | 0.000 | 0.000 | 0.000 | — | 129 |
| slow_burn | 0.000 | 0.500 | 0.167 | 0.020 | 19 | 131 |
| hit_and_run | 0.167 | 0.500 | 0.167 | 0.026 | 19 | 115 |
| signal_evasion | 0.167 | 0.500 | 0.167 | 0.026 | 19 | 137 |
| framing | 0.333 | 0.000 | 0.000 | 0.000 | — | 123 |

Mean over attacks: ASR **0.202**, clean FPR **0.010**, recall **0.119**.
Aggregated over the seven cases, document-level detection has 5 true positives
and 3 false positives (precision 0.625). All three false positives are
`late_clean_3#0000`: it is quarantined in `hit_and_run` and `signal_evasion`,
and flagged but not quarantined in `slow_burn`.
Detection delay is the mean over detected documents only; `—` means no
detection.

### Baseline comparison (mean over the seven attacks)

| Baseline | Mean ASR | Mean clean FPR | Mean recall | Mean LLM calls |
|---|---:|---:|---:|---:|
| no_defence | 0.929 | 0.000 | 0.000 | 29.0 |
| duplicate_filter | 0.929 | 0.000 | 0.000 | 29.0 |
| trust_threshold | 0.929 | 0.000 | 0.000 | 29.0 |
| perplexity | 0.929 | 0.000 | 0.000 | 29.0 |
| TrustRAG | 0.857 | 0.000 | 0.000 | 29.0 |
| RobustRAG | 0.929 | 0.000 | 0.000 | 29.0 |
| always_on_loo | 0.762 | 0.052 | 0.238 | 296.4 |
| **RAGLedger** | **0.202** | **0.010** | **0.119** | **125.9** |

These baselines are implemented against the same policy interface and run on the
same fixture/schedule; they are not the published systems' full pipelines, and
the toy corpus makes the comparison illustrative only.

### Trust-stream smoke (real pipeline, mini corpus)

Command:

```bash
python scripts/run_trust_stream.py --config config/default.yaml \
  --corpus examples/mini_corpus.jsonl --queries examples/mini_questions.json \
  --steps 12 --targets 2 --out runs/trust_stream_check
```

Observed (`runs/trust_stream_check/metrics.json`):

| Metric | Value |
|---|---:|
| poison retrieved rate | 1.000 |
| poison citation rate | 0.000 |
| poison quarantined | 0 |
| clean false quarantine | 0 |
| verifier outcomes | SUPPORT 0, REFUTE 0, NEUTRAL 22 |
| mean LLM calls/query | 4.667 |
| mean latency | 6.75 ms |

The poison was retrieved but never cited, and no clean document was quarantined.
It was also **not detected or quarantined**: with the lexical fallback the only
contradicting copy was a near-duplicate in the same family, which the
independence rule correctly excludes. This is a limitation of the lexical
configuration on this fixture, not evidence that the defence works.

### Demos

| Command | Result |
|---|---|
| `python scripts/demo_end_to_end.py --root /tmp/demo_a_final` | exit 0; 57 chunks indexed; 3 answers; 1 answer flagged after quarantine; no poison retrieved after quarantine |
| `python scripts/demo_person_b.py --root /tmp/demo_b_final` | exit 0; 3 poison docs quarantined; queue drained; later context contains clean documents only |
| `python -m trace_rag.cli --config config/default.yaml doctor` | exit 0; `DEGRADED`, verifier lexical, queue on, trust persistent |
| `python -m trace_rag.cli ... smoke --attacks negation --baselines RAGLedger --targets 1` | exit 0; 1/1 case |

Demo B uses a **scripted** verifier for stage 5 (documented in its output) to
exercise queue → quarantine → remediation without a model. It is a lifecycle
demonstration, not a detection result.

### What was not measured

No Contriever, LLM, DeBERTa NLI, FAISS, full NQ corpus or multi-seed run. No
real ASR, detection FPR, detection delay, remediation recall, latency or cost
numbers. See `docs/REAL_MODEL_EVAL.md`.

---

## G. Limitations

1. **No real-model or full-dataset result.** The only evaluation is the
   synthetic smoke matrix and the mini trust stream. Do not report them as NQ,
   PoisonedRAG or research results.
2. **Lexical verifier false positives.** In `hit_and_run` and `signal_evasion`,
   the correct `late_clean_3#0000` passage was quarantined after two refutations
   from an equal-trust attacker source. The clean FPR is 0.026 in those cases.
   The trust-aware gate prevents a strictly less-trusted witness, but equal
   trust remains ambiguous.
3. **High ASR for negation, instruction injection and framing** (0.33–0.42).
   The cheap signals do not reliably flag these transformations, and the
   lexical verifier often returns NEUTRAL. `instruction_injection` also depends
   on the generator's instruction-following; the stub LLM is not a realistic
   target.
4. **The mini corpus is trivially separable.** All signals can separate the toy
   poison; the scorer demo reports perfect held-out scores that mean nothing.
5. **Optional backends unexercised.** FAISS, HuggingFace embeddings, torch,
   transformers, pyarrow and matplotlib are not installed. Their tests are
   skipped. `run_trust_stream` figures are skipped without matplotlib.
6. **No real-NQ attack/baseline matrix.** `run_chronological_eval.py` is
   fixture-only. `run_trust_stream.py` is the available real-data end-to-end
   path and does not compute ASR.
7. **Suspicion scorer training on real data.** The real runner uses the
   heuristic scorer unless a trained `scorer.joblib` and labelled rows are
   supplied. There is no real-NQ labelled-row generator in this repository.
8. **Single seed.** The smoke matrix uses one seed; no confidence intervals or
   paired significance tests are reported for it. The statistics module exists
   and is unit-tested, but not applied to a multi-seed real experiment.
9. **Historical docs.** `docs/INTERFACES.md`, `docs/HANDOVER*.md` and
   `docs/VERIFICATION.md` retain Person A-era statements and counts. Code,
   README and this report are authoritative; the old documents are retained for
   provenance.
10. **Data/licence.** No NQ corpus or model weights are committed. Check the
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
pytest -o addopts="" -q --cov=trace_rag --cov-report=term
python -m compileall -q src scripts tests

python scripts/check_setup.py          # exit 1 is expected if pyarrow is absent
trace-rag doctor                       # expect DEGRADED in this offline setup

python scripts/run_chronological_eval.py --out runs/smoke_matrix
python scripts/demo_end_to_end.py --root runs/demo
python scripts/demo_person_b.py --root runs/demo_b

python scripts/run_trust_stream.py --config config/default.yaml \
  --corpus examples/mini_corpus.jsonl --queries examples/mini_questions.json \
  --steps 12 --targets 2 --out runs/trust_stream_mini
```

Expected: 235 passed, 6 skipped; 56/56 matrix cases; all three scripts exit 0;
`runs/smoke_matrix/summary.md` carries the explicit smoke-scope statement.

For the real-model run, follow `docs/REAL_MODEL_EVAL.md`. A reportable run
requires `doctor` to say `PROTECTED` with `verifier_mode=nli`,
`trust_persistent=true`, `queue_enabled=true`, and a persistent ledger. Do not
label a `DEGRADED` or `UNPROTECTED` run as RAGLedger.

### Captured evidence

```
baseline_logs/pytest_final.txt
baseline_logs/pytest_cov_final.txt
baseline_logs/compileall.txt
baseline_logs/check_setup_final.txt
baseline_logs/chronological_matrix_final.txt
baseline_logs/demo_end_to_end_final.txt
baseline_logs/demo_person_b_final.txt
baseline_logs/run_trust_stream_final.txt
baseline_logs/doctor_final.txt
runs/smoke_matrix/results.json
runs/smoke_matrix/results.csv
runs/smoke_matrix/summary.md
runs/smoke_matrix/cases/*/stream.jsonl
runs/trust_stream_check/metrics.json
runs/trust_stream_check/stream.jsonl
```

All captured logs are from the environment described at the top of this report.
They are not portable measurements of another machine or GPU.
