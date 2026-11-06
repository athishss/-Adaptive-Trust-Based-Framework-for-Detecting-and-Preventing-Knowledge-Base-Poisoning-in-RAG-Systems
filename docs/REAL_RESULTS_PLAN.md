# Real-results plan: what is measured, against what, and what may be claimed

Status: the **code and notebooks for real-data evaluation are complete and tested offline**.
No real-model result exists yet; nothing in this file is a result. It defines what a result must
look like before it may appear in the paper.

## 1. What is real

| Component | Source | Notes |
|---|---|---|
| Headline attack | PoisonedRAG authors' published `results/adv_targeted_results/nq.json` | Five passages per question injected **verbatim**; MIT licence; pinned SHA in the notebook. |
| Corpus, questions, gold passages | BEIR Natural Questions | Subset built by `scripts/download_data.py --include-queries`; the published questions' gold passages are always kept. |
| Models | Contriever, DeBERTa-v3 NLI, Qwen2.5-7B (Ollama) or Llama-3.1-8B (vLLM) | Revisions/digests recorded in the run manifest. |
| `trustrag_official` | The authors' `defend_module.k_mean_filtering`, unmodified | Stage 1 only; embeddings come from this pipeline (a stated deviation). |
| RobustRAG | Authors' repo, run as published in `ragledger_baselines_official.ipynb` | Own benchmark; **not paired** with the matrix. |

Own attacks (`adaptive`, `slow_burn`, `hit_and_run`, `coordinated`, ...) are real *code* but this project's
own design; they are reported as "own attacks", never as published ones.
Defence rows other than `trustrag_official` are **adaptations**, flagged `published_method: false` in `baselines/policies.py`.

## 2. Metrics (all produced by `scripts/analyze_results.py`)

Unit of analysis is the **target question**; every seed attacks a disjoint block of published questions, so units are independent.
Intervals are 95% percentile bootstrap; tests are exact McNemar on paired binary outcomes with Holm correction per outcome family.

| Metric | Definition | Why it is there |
|---|---|---|
| ASR (first / last attempt) | Attacker answer present, gold absent (whole-word match). Abstention = failure. | Cold-start vs. after-learning protection. |
| Accuracy under attack | Gold answer present in the delivered answer. | A defence that abstains has ASR 0 and accuracy 0; both are shown. |
| Abstention | Share of attempts that abstained. | Same. |
| Exposure | Poison reached the generation context. | Retrieval-level view; not an answer-success rate. |
| Withheld poison / clean rate | Share of retrieved poison / clean docs kept out of the context. | Detection by *withholding* is credited separately from flag-based detection. |
| Flag precision / recall | HIGH band, REFUTE or quarantine. `n/a` when nothing was flagged. | Strict detection; undefined is no longer reported as 0. |
| Benign-control accuracy loss | `none` attack: accuracy vs. `no_defence`, paired. | What the defence costs when nobody attacks. |
| LLM calls per query, p99 latency | From per-case metrics. | Cost. |
| Remediation recall | Delivered poisoned answers later flagged. | The temporal contribution. |

## 3. Hypotheses -> evidence

| Hypothesis (see `RESEARCH_PROTOCOL.md`) | Evidence required |
|---|---|
| H1/H2 efficacy | `poisonedrag_official`, `_sybil`: RAGLedger ASR (first and last) below `no_defence` with a Holm-adjusted p < 0.05 **and** accuracy not below `no_defence`'s by more than the benign-control loss. |
| H3 deferral helps | RAGLedger vs `trust_no_deferred` and `always_on_loo`: ASR no worse, calls per query lower. |
| H4 remediation | Remediation recall on the poisoned answers delivered before quarantine; clean-answer flag rate. |
| H5 adaptivity | `adaptive`, `slow_burn`, `coordinated` (tier `full`): none restores no-defence ASR. |
| H6 cost | Calls/query and p99 from the same runs. |

The strongest differentiator is **temporal**: the published defences are static per query; RAGLedger accumulates source evidence.
That claim is only supportable on the `slow_burn` / `hit_and_run` / repeated-target streams, so those must be in the final run.

## 4. Claims that are NOT allowed

* "Outperforms TrustRAG / RobustRAG" from adapted baselines. Only `trustrag_official` (stage 1) is the authors' code.
* Any number from the smoke tier, the pilot notebook, or a different generator pooled with another.
* ASR without accuracy and abstention next to it.
* RobustRAG numbers in the same significance test as the matrix.
* "Real-world" provenance claims from BEIR (sources are assigned). Use the NELA-GT loader (`scripts/ingest_nela.py`) for that, and state the label file you used.

## 5. Order of execution

1. `ragledger_real_results.ipynb`, tier `smoke`: measures seconds per case on your GPU and exercises every step.
2. Sanity gate (cell 11): undefended exposure and ASR must reproduce the published attack's behaviour for your generator.
3. Tier `main` (3 seeds x 20 questions, 12 defences, 3 attacks incl. the benign control). Resumable.
4. Tier `full` for the temporal and adaptive attacks.
5. `ragledger_baselines_official.ipynb` for RobustRAG as a reference row.
6. Write the paper from `analysis.json` only. Run-time estimates are deliberately not stated: they have not been measured.

## 6. Known limits (state them in the paper)

* The suspicion scorer is trained on this project's template attacks on disjoint real questions; the published poison is unseen. If detection is weak, that is a finding.
* The harness injects poison at a fixed step and repeats each target query (`--target-repeats`); this is a protocol choice, not a natural arrival process.
* One retrieval stack (Contriever, top-5, flat FAISS) and at most two generators.
* NQ only. HotpotQA/MS-MARCO poison files exist in the PoisonedRAG repo and the loader is dataset-agnostic, but they have not been run.
