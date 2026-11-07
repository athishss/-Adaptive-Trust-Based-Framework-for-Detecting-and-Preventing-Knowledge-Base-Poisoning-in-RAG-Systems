# Response to the code-level audit of `real-data-integration` (HEAD 1492403)

Every claim in the audit that was re-checked against the code held up. Status after this change:

| Audit item | Status |
|---|---|
| Q1 resume not identity-bound | **Fixed.** Each case stores a run signature (code HEAD + dirty flag, config bytes, overrides, fixture content, target block, dataset/query/scorer hashes, model digests). Resume reuses only an identical signature. Notebook `OUT`/`BACKUP`/scorer paths also carry a `RUN_TAG`. |
| Q2 baselines forced to lexical verifier | **Fixed.** `build_baseline_verifier` uses the same `security.*` settings as RAGLedger. A deliberate lexical-vs-NLI ablation must now be requested explicitly via `--set security.verifier=...`. |
| Q3 hard sanity gate | **Fixed.** Aborts only on plumbing failure (no eligible attempts, ASR not measurable, poison never in context). Low exposure/ASR prints a warning and is saved in the manifest. |
| Q4 ASR is a string proxy | **Bounded, not solved.** The stream now records `attacker_string_present` separately; `analyze_results.py --audit-sample N` writes a stratified CSV for manual labelling. Report the audited precision of the criterion in the paper. |
| Q5 HIGH-band quarantine skips remediation | **Fixed.** `record_high_band` returns whether it blocked the document; the policy then calls remediation. |
| Q6 `_check_demotion` uses `t_doc` | **Documented, behaviour unchanged.** Switching to `t_eff` would demote every cold-start document capped by `t_cap` and change all results; it is a design decision for the owner. |
| Q7 baseline ledgers never evolve | **Open (by design).** They are labelled adapted static filters; do not describe them as dynamic trust baselines. |
| Q8 `adaptive` not adaptive | **Relabelled** as non-reactive. A closed-loop attacker is still to be built before claiming adaptivity (hypothesis H5). |
| Q9 Qwen is a pilot | **Labelled.** Scope string and manifest carry `PILOT GENERATOR`; never pool with Llama. |
| Q10 no CUDA matmul | **Fixed** in the notebook (fp16 2048x2048 matmul, finite-check). |
| Q11 manifest gaps | **Fixed.** Adds `queries_subset`, scorer and config hashes, run signatures, and faiss/pyarrow/nltk/rouge/vllm/hub versions. |
| Q12 `max_verifier_llm_calls` unused | **Documented** as not enforced (two calls by construction). |
| Q13, Q14, Q15, Q16, Q18, Q19 | **Open**, low/medium; see the audit. |
| Q17 stale docs | README fixed; `RESEARCH_PROTOCOL.md` already carries an update note. |
