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

---

## 5. Rules that keep the experiments honest

1. **Test on Real Data**: The toy corpus in `scripts/demo_person_b.py` is for sanity checks. Run your final evaluations on the BEIR Natural Questions dataset (downloaded via `scripts/download_data.py`).
2. **Clear the DB between runs**: Ensure you are wiping `trust.sqlite3` and `provenance.sqlite3` (or using `:memory:`) between different ablation runs so state doesn't bleed over.
3. **Hardware**: Ensure you run the NLI cross-encoder on a GPU (Colab T4 is fine). Running it on CPU during a 500-query sweep will take hours.

---

## 6. Where to ask

If you need any adjustments to the `TrustLedger` or `CorroborationVerifier` to expose more metrics for your harness, ping Person B. Good luck!
