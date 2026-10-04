# Handover: Person B -> Person C

Written so you (Person C) do not have to assume anything about what Person A and Person B built. If something here disagrees with the codebase, the code wins.

---

## 1. What is finished, and what is not

**Finished and tested by Person A (Base Pipeline & Signals):**
- Provenance tracking, ingestion, and vector indexing (Contriever).
- Trust-weighted retrieval engine.
- Cheap poisoning detection signals (S1-S6) + Logistic Regression Suspicion Scorer.
- Grounded generation with citation tracking and remediation.

**Finished and tested by Person B (Trust Framework & Verification):**
- **TrustLedger (`src/trace_rag/trust/ledger.py`)**: SQLite-backed trust provider using Beta(alpha, beta) distributions.
  - *Source-Inherited Prior*: New docs inherit trust from their source (alpha_0 = kappa * T_source + 1).
  - *T_cap & Decay*: Caps inherited trust until heavily corroborated; naturally decays trust over time.
  - *Burst-aware Prior*: Discounts priors for sources flooding the system.
  - *State Machine*: Automates transitions from TRUSTED -> MONITORED -> QUARANTINED based on T_eff.
- **CorroborationVerifier (`src/trace_rag/trust/verifier.py`)**: 
  - *LOO Influence Proxy*: Lightweight check to see if a claim is redundant (>80% covered by other passages) before spending resources verifying it.
  - *NLI Cross-Encoder*: Uses `cross-encoder/nli-deberta-v3-base` for true semantic entailment/contradiction without generating expensive LLM responses (falls back to lexical overlap if unavailable).
- **TrustPolicy (`src/trace_rag/trust/policy.py`)**: Wires the suspicion bands (LOW, MEDIUM, HIGH) to verification, trust updates, and quarantine remediation callbacks.

**Yours to build (Person C — Attacks, Baselines & Evaluation):**

Your research question is: *Does the defence hold up against attackers who know it exists?*

| WP | Work package | Goal / What it is |
|---|---|---|
| **C1** | PoisonedRAG reproduction | Black-box and white-box attacks on NQ + Contriever, ported into our pipeline. |
| **C2** | Additional attack families | Entity-swap/negation corruption, instruction-injection documents (no query echo). |
| **C3** | Adaptive trust-aware attacks | Hit-and-run (fresh sources), slow-burn (build rep then poison), framing (make true docs look uncorroborated), signal evasion. |
| **C4** | Query-stream simulator | Simulator: clean + target + paraphrased queries, Zipf repetition, gradual injection schedule, simulated contributor provenance, benign ingestion stream from new clean contributors (leakage guard). |
| **C5** | Baselines on the same harness | No defence, perplexity, duplicate filter, TrustRAG, RobustRAG, always-on leave-one-out. |
| **C6** | Metrics + experiment runner | ASR, retrieval P/R/F1, FPR, detection delay, exposure window, remediation recall, cost; 5 seeds, bootstrap CIs, McNemar tests. |

---

## 2. Run it before you write any code

You can run the end-to-end Person B demo to see the Trust framework intercepting a simulated attack in real-time.

```bash
# Ensure dependencies are installed
pip install -e ".[all,data]"

# Run tests to confirm everything works locally (227 tests)
pytest tests/ -q 

# Run the Person B demo simulation
python scripts/demo_person_b.py
```

The output will clearly show the pipeline successfully quarantining "Zog the Alien" passages and returning the correct answer.

### Run it on real data (GPU / Colab)

`docs/COLAB_RUNBOOK.md` (notebook: `notebooks/trace_rag_colab.ipynb`) runs the
same stack on a 200k-passage Natural Questions subset with a real LLM:

```bash
python scripts/run_trust_stream.py --config config/nq_gpu.yaml \
    --corpus data/nq/corpus_subset.parquet --queries data/nq/queries_subset.jsonl \
    --qrels data/nq/qrels.tsv --steps 40 --targets 5 --out runs/nq/trust_stream \
    --set generation.backend=ollama --set generation.model_name=qwen2.5:7b
```

It ingests and indexes the corpus, poisons the passage that carries each target
question's gold answer (with Person C's attack helpers), ingests the poison as a
burst from one fresh source, replays a Zipf query stream through
`PersonAPipeline` with the trust layer wired in, and writes `metrics.json`,
`stream.jsonl`, `trust_history.csv` and the B6 figures.  Read it as the reference
wiring for `TrustLedger` / `CorroborationVerifier` / `TrustPolicy` /
`VerificationQueue` around the pipeline - your experiment runner (C6) needs the
same wiring, over the full matrix instead of one stream.

---

## 3. Map of the Code

Everything you need to interact with is seamlessly integrated into the `PersonAPipeline`.

```text
src/trace_rag/
  trust/
    ledger.py           # Person B's TrustProvider implementation
    verifier.py         # Person B's CorroborationVerifier implementation
    policy.py           # Person B's TrustPolicy implementation
  pipeline.py           # Wires everything together via pipeline.answer()
tests/
  test_trust.py         # 50+ tests covering Trust mechanics. Read these to see how to manipulate state.
```

---

## 4. What you need to implement (Detailed WP Breakdown)

### C1, C2, & C3: Attack Families & Adaptive Attacks
You will need to construct malicious documents to test the robustness of Person A's signals and Person B's Trust Framework.
- **C1**: Implement standard PoisonedRAG.
- **C2**: Create subtle corruptions (negating facts instead of just replacing them) and instruction-injection.
- **C3**: This is where you test Person B's framework to the max. You must try to evade `TrustLedger` by doing a *slow-burn* attack (feeding us good data to build $\alpha$, then attacking) and a *hit-and-run* attack (burning through new sources).

### C4: The Query-Stream Simulator
To accurately evaluate C3 (adaptive attacks), you cannot just run single, isolated queries. You must build a simulator that feeds a chronological stream of queries, benign document ingestions, and malicious document ingestions. This will allow the `TrustLedger` to evolve over time, which is required to measure **Detection Delay** and **Exposure Window**.

### C5: Baselines
Integrate standard industry baselines into your harness so we can compare our framework against them. This includes vanilla RAG (no defence), Perplexity filtering, Duplicate filtering, TrustRAG, RobustRAG, and a naive Always-on LOO (to prove our *Corroboration-gated* LOO saves costs).

### C6: Experiment Runner & Metrics
This will run the simulator (C4) against all attacks (C1-C3) and all baselines (C5).
**Key Metrics to capture per query:**
- **ASR** (Attack Success Rate): Did the LLM output the poisoned fact?
- **FPR** (False Positive Rate): Were clean passages incorrectly quarantined?
- **Cost**: How many LLM calls were made? (`result.llm_calls`)
- **Latency / Exposure Window**: How long did it take the system to retroactively quarantine a malicious source?

You can tune Person B's parameters to find the sweet spots during your sweeps:

```python
from trace_rag.trust.ledger import TrustConfig, TrustLedger

# Example Sweep: Testing the impact of Source-Inherited Priors (kappa)
config = TrustConfig(kappa=3.0, t_cap=0.6)
ledger = TrustLedger("runs/sweep/trust.sqlite3", config=config)
```

### What changed in Person B's components (latest revision)

Person B's trust layer now implements the plan clauses that were missing.  Nothing
Person C already calls changed signature, but three behaviours are different and
matter for the harness:

1. **`verify()` accepts two optional keyword arguments** - `same_burst=` and
   `source_influence=`. `TrustPolicy` passes them from the ledger automatically;
   a bare `verifier.verify(q, qid, doc, pool)` call (the `always_on_loo`
   baseline) behaves exactly as before.  When they are supplied: passages whose
   source arrived in the target's ingestion burst stop corroborating, and
   sources without history contribute less mass.
2. **Verification now costs at most two LLM calls** in both modes: one for the
   single-document claim and at most one for the influence comparison.  The
   lexical fallback no longer summarises every independent passage.
3. **REJECTED is an administrator decision.**  A third refutation on a
   QUARANTINED document now raises a review request instead of rejecting it.
   `ledger.pending_reviews()`, `ledger.admin_approve_rejection(doc_id,
   approved_by=...)`, `ledger.admin_decline_rejection(...)` and
   `ledger.admin_approve_recovery(...)` drive it.  The document stays blocked
   (QUARANTINED) throughout, so security behaviour is unchanged; only the final
   state needs a human.

New knobs on `TrustConfig`: `burst_window_seconds` / `burst_min_docs` (ingestion
bursts are now detected from the ledger's own registration history -
`mark_burst_source` is only an override), `cold_start_influence` /
`source_age_ramp_hours` / `cold_start_min_observations` (Sybil influence cap),
`hierarchical` (False = document-only trust, the V2 ablation) and
`record_history`.  **Trust decay is scheduled per query** (`begin_query`, which
`TrustPolicy.decide` calls once per query), not per observation - a sweep that
bypasses the policy must call `ledger.begin_query()` itself.

High-band passages can be verified off the latency path:

```python
from trace_rag.trust.queue import VerificationQueue

queue = VerificationQueue()
policy = TrustPolicy(ledger=ledger, queue=queue)   # HIGH band is queued, not verified inline
...
policy.drain_verification_queue(verifier)          # between queries, or:
queue.start_background(verifier, ledger, interval=0.5)  # daemon worker
```

Trust dynamics for the B6 figures come straight from the ledger:

```python
from trace_rag.trust.plots import plot_trust_dynamics, plot_quarantine_timeline

history = ledger.export_history()                  # every alpha/beta update
plot_trust_dynamics(history, "runs/figs")          # one PNG per doc/source
plot_quarantine_timeline(ledger.get_audit_log(), "runs/figs/blocked.png")
```

---

## 5. Rules that keep the experiments honest

1. **Test on Real Data**: The toy corpus in `scripts/demo_person_b.py` is for sanity checks. Run your final evaluations on the BEIR Natural Questions dataset (downloaded via `scripts/download_data.py`).
2. **Clear the DB between runs**: Ensure you are wiping `trust.sqlite3` and `provenance.sqlite3` (or using `:memory:`) between different ablation runs so state doesn't bleed over.
3. **Hardware**: Ensure you run the NLI cross-encoder on a GPU (Colab T4 is fine). Running it on CPU during a 500-query sweep will take hours.

---

## 6. Where to ask

If you need any adjustments to the `TrustLedger` or `CorroborationVerifier` to expose more metrics for your harness, ping Person B. Good luck!
