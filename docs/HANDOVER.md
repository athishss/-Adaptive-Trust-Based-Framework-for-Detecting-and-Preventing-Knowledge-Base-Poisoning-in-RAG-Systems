# Handover: Person A → Person B

Written so you do not have to assume anything about what Person A built. If
something here disagrees with the code, the code wins and this file is a bug.

---

## 1. What is finished, and what is not

**Finished and tested** (175 tests, 92% coverage, verified on real BEIR Natural
Questions data with Contriever embeddings on a GPU):

| Plan layer | Where | State |
|---|---|---|
| L0 Ingestion and provenance | `src/trace_rag/ingestion/` | done |
| L1 Trust-weighted retrieval | `src/trace_rag/retrieval/`, `index/`, `embeddings/` | done |
| L2 Six cheap signals | `src/trace_rag/detection/signals.py` | done |
| L3 Calibrated suspicion scorer | `src/trace_rag/detection/scorer.py`, `training.py` | done |
| L5 Answer provenance and remediation | `src/trace_rag/provenance/` | done |
| L7 Grounded generation with abstention | `src/trace_rag/generation/` | done |

**Yours to build** (stubs exist so the pipeline runs without them, and the test
suite already exercises Person A against stand-ins):

| Plan layer | What it is | Protocol to implement |
|---|---|---|
| L4 | Corroboration-gated counterfactual verifier | `Verifier` |
| L6 | Trust ledger and quarantine state machine | `TrustProvider` + `Policy` |

**Not done by anyone yet:** measured results. Attack success rate, detection
rate and the security/cost trade-off need your ledger plus Person C's attacks.
Nothing in this repo claims a research result.

---

## 2. Run it before you read any code

```bash
conda env create -f environment.yml && conda activate trace-rag
pip install -e ".[all,data]"
pytest -q                            # 175 tests, ~15 s
python scripts/check_setup.py        # what is installed, GPU visible?
python scripts/demo_end_to_end.py    # the whole pipeline on a bundled toy corpus
```

The demo ingests a clean corpus, injects PoisonedRAG-style passages from a new
contributor, answers the attacked question before and after training the
detector, then quarantines the injected passages and shows earlier answers being
flagged. Five minutes with this gives you the whole mental model.

On real data:

```bash
python scripts/download_data.py --dataset nq --out data/nq --subset 200000
trace-rag --config config/nq_gpu.yaml ingest-beir --corpus data/nq/corpus_subset.parquet
trace-rag --config config/nq_gpu.yaml index          # resumable; --limit N to work in sittings
trace-rag --config config/nq_gpu.yaml query "who designed the eiffel tower?" --set generation.backend=stub
```

`--set key=value` overrides any config value without editing a file, before or
after the subcommand.

---

## 3. Map of the code

```
src/trace_rag/
  contracts.py          THE INTEGRATION SURFACE - import from here, nothing else
  config.py             every threshold; nothing is hard-coded
  pipeline.py           PersonAPipeline: wires the layers, exposes answer()
  cli.py                the trace-rag command

  ingestion/
    parsers.py          PDF, TXT, Markdown, HTML, BEIR Parquet and JSONL
    chunking.py         ~100-word windows matching BEIR passage size
    provenance.py       SQLite store: sources, documents, chunks, burst counts
    pipeline.py         Ingestor + simulated contributor assignment
  embeddings/
    hashing_embedder.py deterministic, offline - tests and development only
    hf_embedder.py      Contriever / BGE (mean or CLS pooling, L2-normalised)
  index/
    numpy_index.py      exact cosine; the oracle FAISS is tested against
    faiss_index.py      flat / IVF-PQ / HNSW, with tombstones on re-add
  retrieval/
    retriever.py        score = similarity * max(t_eff, floor) ** lambda
  detection/
    signals.py          S1..S6, no LLM calls
    scorer.py           logistic regression + isotonic calibration + thresholds
    training.py         leakage-guarded dataset building, splits, LOAO
  generation/
    prompts.py          every prompt used, in one file
    llm.py              stub / vLLM / Ollama / transformers backends
    citations.py        citation parsing, trust-weighted evidence mass
    grounded.py         cited answers, abstention
  provenance/
    answer_log.py       which passages produced which answer
    remediation.py      quarantine -> flag every past answer that used it
```

Docs: `INTERFACES.md` (how to plug in), `REQUIREMENTS.md` (requirement → test),
`VERIFICATION.md` (16 defects found and fixed, and what is still unverified),
`RUNNING_ON_GPU.md` (conda, CUDA, Windows).

---

## 4. What you receive, per query

```python
result = pipeline.answer("who designed the eiffel tower?", query_id="nq_12")

result.retrieval.documents   # top-k RetrievedDocument
result.retrieval.pool        # the full 50-passage candidate pool  <- you need this
result.assessments           # SecurityAssessment per passage: signals, suspicion, band
result.decision              # PolicyDecision - yours, once you implement Policy
result.record                # AnswerRecord: answer, citations, abstained, evidence mass
result.llm_calls             # generation + verification, summed
```

Each `RetrievedDocument` carries `doc_id`, `chunk_id`, `text`, `similarity`,
`rank`, `source_id`, `family_id`, `trust` (a `TrustSnapshot`) and `metadata`
with `parent_doc_id`, `ingested_at`, `sha256`, `n_words`.

**`doc_id` is the chunk id** (for example `nq_1#0000`), because trust is tracked
per passage. The parent document is in `metadata["parent_doc_id"]`.

---

## 5. What you implement

### TrustProvider — the read side of your ledger

```python
class TrustLedger:
    def get_trust(self, doc_ids: Sequence[str]) -> Dict[str, TrustSnapshot]: ...
    def blocked_doc_ids(self) -> set: ...     # QUARANTINED + REJECTED
```

`TrustSnapshot` fields: `t_doc`, `t_family`, `t_source`, `t_eff`, `status`,
`n_doc_observations`, `as_of`. `t_eff` is what re-ranks retrieval. Called once
per query with up to 50 ids — batch the SQLite read.

### Verifier — corroboration-gated counterfactual verification

```python
class CorroborationVerifier:
    def verify(self, query, query_id, target, pool) -> VerificationResult: ...
```

Called only for MEDIUM-band passages. `pool` is the full candidate pool, so you
can pick corroborating passages whose `source_id` **and** `family_id` differ
from the target's. Reuse Person A's prompt for the single-document step:

```python
from trace_rag.generation.prompts import build_single_doc_prompt
```

Set `llm_calls` honestly — Person C reports mean LLM calls per query and sums it
from your result.

### Policy — escalation and quarantine

```python
class TrustPolicy:
    def decide(self, query, query_id, documents, assessments, verifier=None) -> PolicyDecision: ...
```

Return what reaches the generator, what you dropped, and every
`VerificationResult` you produced. `contracts.DefaultPolicy` is a working
band-only version — read it as the reference.

### Wiring

```python
from trace_rag import Config, PersonAPipeline

pipeline = PersonAPipeline.from_config(Config.load("config/nq_gpu.yaml"))
pipeline.trust_provider = ledger
pipeline.retriever.trust = ledger      # retrieval reads trust directly
pipeline.verifier = verifier
pipeline.policy = policy
```

### Quarantine → remediation

```python
report = pipeline.on_quarantine([doc_id], reason="two independent refutations")
# report.newly_flagged, report.exposure_window, report.affected_answer_ids
```

Pass `dry_run=True` to preview the blast radius without writing.

---

## 6. Rules that keep the experiments honest

Breaking any of these invalidates the results, so they are enforced in code
where possible:

1. **Person A never writes trust.** Only your verified outcomes update the
   ledger. Cheap signals decide what to *check*, never what to trust — that is
   what stops the detector reinforcing its own false positives.
2. **Corroboration counts independent sources**, not passages: different
   `source_id`, different `family_id`, not the same ingestion burst. Otherwise
   five passages from one attacker corroborate each other.
3. **Features are snapshots.** `TrustSnapshot.as_of` is the retrieval moment;
   `LabelledRow.validate()` raises `LeakageError` if a snapshot post-dates its
   query.
4. **Splits are by question**, never by row (`split_by_question`), and
   thresholds and calibration come from validation only.
5. **Labels never enter the pipeline.** `is_poison` is a parameter of the
   training helper; nothing at inference time reads an attack label.

---

## 7. What Person A guarantees, and what you should not assume

Guaranteed: `contracts.py` is versioned (`CONTRACT_VERSION`) and will not change
shape without telling you. Retrieval is deterministic for a fixed config and
seed. Quarantined passages never reach retrieval, context or a citation — there
are tests for exactly that. Ingestion, indexing and remediation are idempotent.
Indexing is resumable.

Do not assume: that `doc_id` is a document (it is a chunk); that trust defaults
to anything other than 0.5 before your ledger exists; that the bundled demo
corpus means anything (it is trivially separable and says so); that the numbers
in `VERIFICATION.md` were measured on a GPU cluster (they are from a laptop).

---

## 8. Suggested order for your work

1. Run the demo and read `contracts.py` end to end (one sitting).
2. Write a ten-line `TrustProvider` returning fixed values, assign it, run
   `pytest -q`, and query something. This proves the seam in under an hour.
3. Build the ledger properly: document, family and source Beta(α, β), the
   empirical-Bayes blend, bounded asymmetric updates, decay, cold-start caps.
4. Build the verifier: single-document answer, leave-one-out influence, NLI
   corroboration against independent sources, trust-weighted support/refute.
5. Build the policy and quarantine state machine; call
   `pipeline.on_quarantine(...)` from it.
6. Add your own tests mirroring `tests/test_pipeline_integration.py`, which
   already shows how to test against Person A with stand-ins.

---

## 9. Where to ask

Open an issue on the repo and tag Person A. If something in this handover is
wrong or missing, say so — it is meant to make assumptions unnecessary.
