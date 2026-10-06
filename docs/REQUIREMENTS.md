# Requirements audit — Person A

Every Person A requirement from the plan, where it is implemented, what proves
it, and its honest status. "Done" means implemented **and** covered by a test
that fails if it breaks. Nothing here claims a research result; those need
Person C's attacks and real data.

## Work packages (plan §9, Person A)

| WP | Requirement | Implementation | Evidence | Status |
|---|---|---|---|---|
| A1 | Ingestion & provenance: PDF/TXT/MD/HTML, ~100-word chunks, SHA-256, source/version metadata, near-duplicate families | `ingestion/parsers.py`, `chunking.py`, `provenance.py`, `utils/hashing.py` | `test_ingestion.py` (17 tests), collision fuzzing in `test_regressions.py` | **Done** |
| A2 | Retrieval at scale: Contriever/BGE embeddings, FAISS on full NQ, trust-weighted re-ranking | `embeddings/`, `index/` (numpy, FAISS flat/IVF-PQ/HNSW), `retrieval/retriever.py` | `test_index_and_retrieval.py` (12), numpy↔FAISS parity, IVF-PQ and HNSW round-trips, real transformer embedder in `test_model_backends.py` | **Done**, not yet run on full NQ |
| A3 | LLM serving + grounded generation: mandatory citations, NLI citation check, abstention | `generation/` (prompts, 4 backends, citations, grounded) | `test_generation.py` (15), `test_model_backends.py` (served-model path, request shape, error handling) | **Done** |
| A4 | Six cheap signals, no LLM calls | `detection/signals.py` | `test_signals.py` (15), including "signals cost no LLM call" and per-signal behaviour | **Done** |
| A5 | Suspicion scorer: logistic regression + isotonic calibration, thresholds from validation, leave-one-attack-family-out | `detection/scorer.py`, `detection/training.py` | `test_scorer_and_training.py` (15), constraint tests, leakage guards, LOAO disjointness in `test_regressions.py` | **Mechanism done**; the LOAO *result* needs Person C's attack families |
| A6 | Answer-provenance log + retroactive remediation | `provenance/answer_log.py`, `remediation.py` | `test_provenance_log.py` (9), quarantine→flag path in `test_pipeline_integration.py` | **Done** |
| A7 | Latency profiling + signal ablation | `scripts/profile_and_ablate.py` | `test_regressions.py::test_profile_and_ablate_script`; measured numbers in `VERIFICATION.md` | **Done**; numbers exclude real embedding/LLM time |

## Person A definition of done (plan §9.4)

| Requirement | Status |
|---|---|
| Full-NQ retrieval with trust re-ranking | Code and config ready (`config/full_nq.yaml`), **not yet run on NQ** — no dataset access in the build environment |
| Six signals + calibrated scorer with leave-one-attack-family-out results | Signals, scorer and LOAO machinery done and tested; **results pending** Person C's attacks |
| Cited, abstaining generation | Done |
| Remediation flags past answers | Done |
| Latency profile | Done (offline backends; rerun with Contriever + Llama for the report) |

## Design requirements from the plan

| Requirement | Where | Status |
|---|---|---|
| Trust is read, never written by Person A | `retrieval/retriever.py`, `pipeline.py` | Done — no write path exists |
| Cheap signals never quarantine on their own | `contracts.DefaultPolicy`, bands | Done |
| Trust-weighted retrieval `sim × t_eff^λ` with cold-start floor | `retrieval/retriever.py` | Done, floor tested |
| Quarantined passages excluded from retrieval | `retriever`, `TrustProvider.blocked_doc_ids` | Done, tested end to end |
| Evidence aggregated by source, not by passage | `generation/citations.py` | Done, tested |
| Thresholds satisfy FPR cap **and** escalation budget | `scorer.select_thresholds` | Done, tested on hard data |
| Features snapshotted at retrieval time (no future information) | `contracts.TrustSnapshot`, `training.LabelledRow.validate` | Done, `LeakageError` tested |
| Splits by question, thresholds from validation only | `training.split_by_question`, `train_scorer` | Done, tested |
| Leakage guard: benign late arrivals in the corpus | demo corpus + `docs/INTERFACES.md` rule | Done for the demo; Person C must honour it on real data |
| Everything configurable, nothing hard-coded | `config.py`, `config/*.yaml` | Done, fuzzed over random valid configs |
| Deterministic given a seed | seeds, sorted tie-breaks, stable ids | Done, verified across processes |

## Integration requirements (Person B and C)

| Requirement | Status |
|---|---|
| One import surface, versioned (`CONTRACT_VERSION`) | Done — `trace_rag/contracts.py` |
| B's three protocols with working defaults to match | Done — `NullTrustProvider`, `NullVerifier`, `DefaultPolicy`, all protocol-tested |
| B's components exercised against the pipeline | Done — stand-in ledger/verifier/policy in `test_pipeline_integration.py` |
| C drives the pipeline and gets JSON-serialisable results | Done — `answer()`, `answer_stream()`, `to_dict()` |
| C's labels never read by the pipeline | Done by construction — `is_poison` is a parameter of the training helper |
| Ablations are config edits, not code edits | Done — `signals.enabled`, trust/abstention knobs |

## Panel questions that touch Person A

| Question | Answered by |
|---|---|
| Q1 methodology and framework | The eight layers, `README.md` architecture table |
| Q2 knowledge base | BEIR corpora (Parquet or JSONL) + the provenance store |
| Q3 dataset for training/validation/evaluation | `detection/training.py`: labels from the harness, split by question, LOAO |
| Q4 which LLM and why | `config/full_nq.yaml` + four interchangeable backends |
| Q5 document formats | PDF/TXT/MD/HTML and BEIR Parquet/JSONL, all tested |

## Not done, and why

* **No real dataset has run through this code in the current sandbox.** NQ
  downloads, end-to-end subset evaluation, and benchmark outputs still need to
  be executed from the Colab notebook/runbook. The readers and subset logic are
  tested offline against generated files; this is not a substitute for running
  the published files.
* **No research result.** Detection accuracy, attack success rate and the
  security/cost trade-off need the complete Person C matrix, real data, multiple
  seeds, and confidence intervals. The mini stream is a plumbing regression
  only.
* **No live accelerator/model validation here.** TPU/XLA embedding and NLI
  support is implemented and unit-tested with device/API mocks, and local HF
  generation can select XLA, but this environment has no torch, transformers,
  CUDA, or torch_xla. Contriever, DeBERTa NLI, and a real answer model have not
  been run together on a GPU or TPU. Do not present the TPU path as validated
  until `notebooks/trace_rag_colab_tpu.ipynb` completes on a real TPU runtime.
