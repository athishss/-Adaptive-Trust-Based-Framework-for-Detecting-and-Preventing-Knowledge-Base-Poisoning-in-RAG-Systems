# Verification pass

After the first working version, the code was probed adversarially rather than
just re-run against its own tests. Six defects were found and fixed; each one
now has a regression test in `tests/test_regressions.py`.

| # | Defect | Why it mattered | Fix | Test |
|---|---|---|---|---|
| 1 | **FAISS kept the old vector** when a passage id was re-indexed after its content changed (numpy overwrote correctly, FAISS silently skipped) | Person C injects and updates passages mid-stream; retrieval would have scored the stale text while the store returned the new one | positions are tombstoned and the new vector wins; `dead_rows` exposes how many tombstones exist, and they survive save/load | `test_reindexing_changed_content_updates_the_vector`, `test_faiss_tombstones_are_not_returned_and_survive_reload` |
| 2 | **Leave-one-attack-out leaked clean rows**: the same clean rows (and questions) sat in train and test | the held-out false-positive rate would have been measured on passages the detector trained on — a reviewer would catch this | clean rows are split by question; every row of a question the held-out family attacked goes to the test side; other families' rows on those questions are dropped | `test_leave_one_attack_out_shares_no_rows_or_questions` |
| 3 | **Quadratic ingestion**: source counters were recomputed with `COUNT(*)` per document | 2.68M-passage ingestion would not have finished; measured 139 passages/s and falling | counters are maintained incrementally, and bulk loads share one transaction (`store.batch()`) | `test_source_counters_stay_correct_through_reingestion`, `test_ingestion_of_a_thousand_passages_is_not_quadratic` |
| 4 | **Unbounded near-duplicate cost**: LSH buckets grew without limit, so ingestion degraded superlinearly (578/s at 5k → 139/s at 20k) | same as above, plus ~9 GB of RAM extrapolated to full NQ | per-insert work is capped (`max_candidates`, `max_bucket`), and `family_backend: exact` is available for full-corpus runs | `test_minhash_candidate_cap_keeps_assignment_deterministic`, `test_family_backends` |
| 5 | **IVF-PQ training guard was wrong**: it raised on corpus sizes FAISS handles and stayed silent about the PQ codebook size | full-corpus indexing would either fail or quietly produce a badly trained index | raises only when clustering is impossible (`n < nlist`), warns with the recommended count otherwise | `test_ivfpq_refuses_impossible_training_size`, `test_faiss_ivfpq_and_hnsw_round_trip` |
| 7 | **The loader could not read the files we told the team to download**: Hugging Face serves BEIR as Parquet, the ingestor only read JSONL | the documented download would have failed at the first command | `iter_beir_corpus` streams Parquet (row-group batches) and JSONL, and rejects anything else loudly | `test_beir_parquet_is_readable`, `test_beir_parquet_without_id_column_fails_loudly` |
| 8 | **Work package A7 had no runnable entry point**: latency profiling and signal ablation were possible but not scripted | A7 is a deliverable; "supported by config" is not the same as delivered | `scripts/profile_and_ablate.py` reports per-stage latency percentiles, LLM calls per query, and held-out AUC with each signal removed | `test_profile_and_ablate_script` |
| 6 | **Empty `[]` brackets survived** citation cleanup | cosmetic, but it leaked model formatting into logged answers | cleanup strips empty brackets too | `test_empty_and_invalid_brackets_are_cleaned` |

## Second pass (static analysis, fuzzing, concurrency, model backends)

| # | Defect | Why it mattered | Fix | Test |
|---|---|---|---|---|
| 9 | **Chunk ids collided**: ids were sanitised and truncated to 48 characters with no hash, so `doc/one` and `doc one`, or two long ids sharing a prefix, produced the same chunk id | `INSERT OR REPLACE` then silently overwrote one document's passage with another's — data loss and wrong provenance, most likely with file paths and URLs as ids | ids that need sanitising or exceed 48 characters get a hash of the full original after a `~`, which the sanitiser can never produce; benchmark ids stay readable | `test_chunk_ids_do_not_collide`, `test_chunk_ids_are_unique_under_fuzzing` (5k adversarial ids), `test_colliding_document_ids_keep_separate_rows` |
| 10 | **`torch_dtype=` is deprecated** in transformers v5 | the embedder and local-LLM paths would break on a current install | both spellings are tried | `test_hf_embedder_*` |
| 11 | **`HFLocalLLM` required `accelerate`** because it always passed `device_map=` | a plain single-GPU or CPU run failed with a confusing dependency error | `device_map` is only used when explicitly asked for (`auto`/`balanced`/`sequential`); otherwise the model is moved with torch alone | `test_hf_local_llm_generates_from_a_locally_built_model` |

Static analysis (pyflakes, ruff `F,E9,B,SIM,RUF`) found no logic defects; unused
imports were removed and `zip()` calls whose lengths must match are now
`strict=True`.

## Third pass (scale and real-data path)

| # | Defect | Why it mattered | Fix | Test |
|---|---|---|---|---|
| 12 | **Indexing loaded the whole store into memory** (`list(iter_chunks())`) | a 2.68M-passage corpus would have needed several GB of Python objects before the first vector was computed | passages are streamed from SQLite in batches; memory stays flat (measured: 40k passages indexed with an 86 MB rise, which is the index itself) | `test_indexing_streams_instead_of_loading_everything` |

Added in the same pass: `scripts/download_data.py` (resumable Hugging Face
download plus a subset builder that never drops a gold passage), `environment.yml`,
`config/nq_gpu.yaml` and `docs/RUNNING_ON_GPU.md`.

## Fourth pass (first real-data run)

Running the pipeline over real Natural Questions passages on a Windows laptop
with an RTX 3050 Ti exposed two defects that toy data could not.

| # | Defect | Why it mattered | Fix | Test |
|---|---|---|---|---|
| 13 | **Signal S2 fired on everything.** It compared a passage's similarity against the whole candidate pool, but signals are only computed for the top-k, which are the highest similarities in that pool by construction. On real retrieval, ordinary clean passages scored 0.81-0.98. | the signal carried no information and inflated suspicion for every passage, pushing clean content into the MEDIUM band and wasting the verification budget | S2 now measures detachment from the similarity curve: the largest gap in the top of the pool against the typical gap there, flagging only what sits above an unusually large gap. Clean smooth decay scores 0.07, a cluster of injected passages scores 1.0 | `test_similarity_outlier_is_near_zero_on_a_smooth_decay`, `test_similarity_outlier_is_robust_to_several_injections` |
| 14 | **Indexing had no checkpoints.** A power cut twelve hours into a 200k run lost everything. | unusable on a laptop, and painful on any machine | `index_chunks` skips passages already in the index, checkpoints every `--save-every` passages and honours `--limit`, so a corpus can be indexed in sittings | `test_indexing_resumes_and_skips_what_is_done`, `test_index_limit_lets_you_work_in_sittings` |

Measured on that laptop: ingestion of 5,000 BEIR passages in seconds, Contriever
embedding and FAISS indexing of 6,128 passages in about 66 seconds including
model load, and retrieval in 316 ms. Every passage returned for "who designed
the eiffel tower?" was about the Eiffel Tower.

## Fifth pass (edge cases and failure modes)

A sweep of the failure modes a real deployment hits. Eleven of thirteen passed
unchanged; two produced unhelpful errors and were fixed.

| # | Defect | Fix | Test |
|---|---|---|---|
| 15 | A truncated or corrupted index file failed with `UnpicklingError`, which does not say what to do | `IndexLoadError` naming the file and the command that rebuilds it; a genuinely missing file still raises `FileNotFoundError` | `test_corrupt_numpy_index_explains_itself`, `test_corrupt_faiss_index_explains_itself` |
| 16 | A database that could not be opened reported SQLite's bare message with no path | the error names the store and the path, and points at `storage.root` | `test_unopenable_store_names_the_path` |

Passed unchanged: a corpus smaller than `top_k`; every document quarantined
(abstains cleanly); index/store drift after rows vanish; an invalid config; an
empty corpus file; re-ingesting the same corpus twice (idempotent); querying
while another thread indexes; a corrupt scorer file; a 50,000-word document and
a passage of pure punctuation; a trust provider that raises (the error surfaces
rather than being swallowed); two pipelines sharing one database.

Not testable in this container: file-permission failures, because it runs as
root and root bypasses permission checks.

## Checks that passed first time

Fuzzing (300 hostile documents, 120 random queries, 40 random configurations)
with invariants checked on every answer: signals in [0,1], evidence mass in
[0,1], no answer without a citation, no citation of a passage that was not
retrieved or that policy excluded. Six threads answering concurrently against
one pipeline, and both databases surviving a reopen. Empty index and empty
corpus, unicode and CJK text, a 2000-word query, very long
single tokens, duplicate document ids, concurrent ingestion from four threads,
threshold constraints on hard (overlapping) data, calibration error below 0.15
in every populated bin, rejection of wrong feature counts, the leakage guard on
post-query feature snapshots, adversarial citation strings (`[../etc/passwd]`,
stacked brackets), evidence mass that cannot be inflated by repeats from one
source, and identical output across separate processes.

## Measured cost (this container: 2 vCPU, hashing embedder)

| Operation | Rate | Extrapolated to NQ (2.68M passages) |
|---|---|---|
| Ingestion, `family_backend: exact` | ~13,000 passages/s | ~3.5 min, flat memory |
| Ingestion, `family_backend: minhash` | ~2,200 passages/s | ~20 min, ~5 GB RAM |
| Query (retrieval + signals + scoring + generation) | 2.7 ms mean, 3.6 ms p95 | signals dominate (~1.8 ms) |

Embedding and the real LLM are not in those numbers — they are the actual cost
on GPU and must be measured again with Contriever and Llama-3.1-8B.

## Sixth pass (trust false positives, artifact integrity, and TPU paths)

| Defect / gap | Fix | Regression evidence |
|---|---|---|
| A same-topic passage with no assertion of the query's relation (for example, Eiffel Tower height text used against a "who designed" claim) could be treated as a lexical refutation | Lexical refutation now requires the question predicate family to occur in the independent passage when one is identified | `test_same_subject_without_the_question_relation_is_not_refutation`; 12-step mini stream now quarantines both injected poisons with zero clean false quarantines |
| A numeric answer could be falsely supported by shared units/topic words, or a numeric poison could be replaced with an entity answer | Numeric comparisons require matching number values; stream poison helpers accept type-compatible false answers and generate numeric fallbacks | `test_numeric_disagreement_is_not_support_from_a_shared_unit`, `tests/test_run_trust_stream.py` |
| Repeated refutations could mature a new source's cold-start influence | Only positive/support observations count toward the support-history maturity threshold | `test_refutations_do_not_mature_new_source_influence` |
| Explicit `nli_mode: nli` could silently run lexical comparison when model loading failed | Explicit NLI requests now fail fast; the actual verifier mode/device and resolved config are saved with stream artifacts | `test_explicit_nli_mode_does_not_silently_fall_back`, `test_mini_stream_writes_reproducible_metrics_without_clean_quarantine` |
| NLI input was constructed as one string rather than an explicit premise/hypothesis pair | Both the Transformers pipeline and XLA path now feed paired inputs | `test_nli_pipeline_receives_premise_and_hypothesis_as_a_pair` |
| S6 assumed the target passage was the first returned neighbour | Signal computation passes the target id and excludes it by ID, independent of ANN ordering/ties | `test_neighbourhood_density_excludes_target_by_id_not_neighbour_rank` |
| The HF device path supported CPU/CUDA only | Added PyTorch/XLA device resolution, static TPU embedding shapes, TPU NLI inference, optional local HF generation device selection, `config/nq_tpu.yaml`, and a Colab notebook | `tests/test_device_resolution.py` and config tests use mocks; **no actual TPU execution has occurred here** |

`pytest -q -ra` passed in the development sandbox; six optional tests were
skipped because FAISS, torch/Transformers, and Matplotlib were absent. Python
`compileall`, `git diff --check`, and the setup check passed. The CPU mini smoke
used the lexical verifier and `StubLLM`: 2 poison documents quarantined, 0 clean
false quarantines, 0 poison citations, poison retrieval rate 0.5. This is a
regression fixture only—not an accuracy or research result.

## Current unverified items

* **Live TPU/XLA behavior.** The available tests mock XLA APIs. The sandbox has
  no `torch`, `transformers`, `torch_xla`, CUDA, or TPU. Run
  `notebooks/trace_rag_colab_tpu.ipynb` on Colab and retain its device, timing,
  model, seed, and resolved-config metadata before claiming TPU execution.
* **Real weights and data.** Contriever, the DeBERTa NLI model, and an answer
  model have not been run together against published NQ files in this session.
  The optional Colab NQ cell downloads source data and prepares a subset, but it
  has not been executed here.
* **Research evaluation.** The mini stream is not representative. The callback
  matrix is not a complete controlled attack × baseline × seed benchmark, and
  no ASR, confidence interval, baseline comparison, or headline result is
  established by these tests.
* **Scale and enterprise operations.** Full multi-million-passage scale,
  deployment security/privacy, tenant isolation, SLOs, operational monitoring,
  backup/restore, and company-specific policy have not been validated. See
  `docs/PRODUCTION_READINESS.md`.
