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

## Checks that passed first time

Empty index and empty corpus, unicode and CJK text, a 2000-word query, very long
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

## What is still unverified

* `HFEmbedder` (Contriever/BGE) and the vLLM / Ollama / transformers generation
  backends: no GPU or model weights in the build environment. The code paths are
  import-guarded and written to the documented APIs, but run each once on your
  own machine before trusting them.
* Behaviour at true corpus scale (millions of passages) is extrapolated from
  20k-passage measurements, not observed.
* The bundled mini corpus is trivially separable; no accuracy claim should come
  from it. The A7 ablation makes this visible: every signal can be removed with
  no AUC loss, because any single one separates the toy poison.
* **No real dataset has been through this code.** The build environment has no
  internet egress to Hugging Face, so NQ/HotpotQA/MS-MARCO were never
  downloaded. The Parquet and JSONL readers are tested against files written to
  the exact published schema, not against the published files themselves.
