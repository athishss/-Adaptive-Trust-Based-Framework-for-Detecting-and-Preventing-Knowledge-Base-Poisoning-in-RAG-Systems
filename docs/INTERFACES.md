# Integration guide (Person B and Person C)

Everything you need is in `trace_rag.contracts`. Import from there, not from
internal modules — that file is versioned (`CONTRACT_VERSION`) and is the only
thing Person A promises not to break without telling you.

```python
from trace_rag.contracts import (
    RetrievedDocument, SecurityAssessment, VerificationResult, PolicyDecision,
    TrustSnapshot, TrustStatus, Band, VerificationOutcome, AnswerRecord,
    TrustProvider, Verifier, Policy,     # protocols to implement
)
```

## What Person A gives you

| Object | Produced by | Contains |
|---|---|---|
| `RetrievedDocument` | `pipeline.retrieve()` | `doc_id`, `chunk_id`, `text`, `similarity`, `rank`, `source_id`, `family_id`, `trust` (snapshot), `metadata` (`ingested_at`, `parent_doc_id`, `sha256`, `n_words`) |
| `RetrievalOutcome.pool` | `pipeline.retrieve()` | the full candidate pool (K′ = 50), which the verifier needs for corroboration |
| `SecurityAssessment` | `pipeline.assess()` | six signals, calibrated `suspicion`, `band` (LOW/MEDIUM/HIGH), `action`, full feature snapshot |
| `AnswerRecord` | `pipeline.answer()` | answer, citations, used/excluded passages, trust snapshot per passage, evidence mass, LLM calls, latency |

`doc_id` is the **chunk id** (for example `nq_1#0000`). Trust is tracked at that
level. The parent document is in `metadata["parent_doc_id"]`.

## Person B: three protocols to implement

### 1. `TrustProvider` — the trust ledger, read side

```python
class TrustLedger:
    def get_trust(self, doc_ids: Sequence[str]) -> Dict[str, TrustSnapshot]:
        ...   # one snapshot per doc_id, t_eff in [0, 1]
    def blocked_doc_ids(self) -> set:
        ...   # QUARANTINED + REJECTED; these never get retrieved
```

`TrustSnapshot` fields: `t_doc`, `t_family`, `t_source`, `t_eff`, `status`,
`n_doc_observations`, `as_of`. Person A only reads them; it never writes trust.
`t_eff` is what re-ranks retrieval: `score = similarity * max(t_eff, floor) ** lambda`.

Keep `get_trust` fast — it is called once per query with up to 50 ids. Batch the
SQLite read.

### 2. `Verifier` — corroboration-gated counterfactual verification

```python
class CorroborationVerifier:
    def verify(self, query: str, query_id: str,
               target: RetrievedDocument,
               pool: Sequence[RetrievedDocument]) -> VerificationResult:
        ...
```

Called only for MEDIUM-band passages. `pool` is the candidate pool, so you can
pick corroborating passages whose `source_id` **and** `family_id` differ from the
target's. Set `llm_calls` honestly — Person C reports mean LLM calls per query
and it is summed from your result.

For the "what does this passage alone claim" step, reuse Person A's prompt:

```python
from trace_rag.generation.prompts import build_single_doc_prompt
```

### 3. `Policy` — escalation and quarantine

```python
class TrustPolicy:
    def decide(self, query, query_id, documents, assessments, verifier=None) -> PolicyDecision:
        ...
```

Return the passages that reach the generator (`context_doc_ids`), the ones you
dropped (`excluded_doc_ids`) and every `VerificationResult` you produced.
`trace_rag.contracts.DefaultPolicy` is a working band-only implementation; use it
as the reference.

### Wiring it up

```python
from trace_rag import Config, PersonAPipeline

pipeline = PersonAPipeline.from_config(Config.load("config/full_nq.yaml"))
pipeline.trust_provider = ledger          # your TrustProvider
pipeline.retriever.trust = ledger         # retrieval reads trust directly
pipeline.verifier = verifier
pipeline.policy = policy
```

### Quarantine → remediation

When your state machine quarantines a passage, call:

```python
report = pipeline.on_quarantine([doc_id], reason="two independent refutations")
# report.newly_flagged, report.exposure_window, report.affected_answer_ids
```

Every earlier answer that used that passage is flagged in the answer log. Pass
`dry_run=True` to preview without writing.

### NLI checker (optional but useful)

If you expose your NLI cross-encoder as `checker(premise, hypothesis) -> (bool, float)`,
pass it in as `citation_checker=` and Person A will verify every cited sentence
against its passage and record citation precision.

## Person C: driving experiments

```python
result = pipeline.answer(query, query_id="nq_12", step=1532)
result.record          # AnswerRecord: answer, citations, abstained, evidence_mass
result.assessments     # per-passage suspicion, band, signals
result.llm_calls       # generation + verification calls for this query
result.timings_ms      # retrieval / signals_and_scoring / policy / generation / total
result.to_dict()       # JSON-serialisable, one line per query in results.jsonl
```

Run a whole stream (`stream.jsonl` rows: `{"t", "qid", "text"}`):

```python
results = pipeline.answer_stream(rows)
```

**Labels stay on your side.** `is_poison` is passed *into* the training helper;
the pipeline never reads a label file:

```python
from trace_rag.detection import build_rows, TrainingSet, train_scorer

dataset = TrainingSet()
outcome = pipeline.retriever.retrieve(query, qid)
snapshots = pipeline.signals.compute(query, outcome.documents, outcome.pool)
dataset.extend(build_rows(qid, outcome.documents, snapshots,
                          is_poison=lambda d: d in poison_ids,
                          attack_family="poisonedrag_bb"))
scorer, thresholds, held_out = train_scorer(dataset)   # split by question, val-only thresholds
```

Ablations are config edits, not code edits:

| Variant | How |
|---|---|
| V0 no defence | `retrieval.trust_lambda: 0`, `scorer.theta_high: 1.01`, no verifier/policy |
| V1 signals only | trained scorer, `NullTrustProvider` |
| V2–V4 | plug in Person B's ledger / verifier |
| single-signal ablation | `signals.enabled: {s3: false}` |
| no abstention | `generation.abstain_evidence_mass: 0.0` |

Metrics Person A already computes for you: `pipeline.answer_log.stats()`
(abstention rate, mean LLM calls, mean latency), `RemediationService.remediation_recall(ids)`,
and `report.exposure_window` per quarantine event.

## Rules that keep the experiments honest

1. Features are snapshotted at retrieval time. `LabelledRow.validate()` raises
   `LeakageError` if a snapshot post-dates its query.
2. Splits are by question, never by row (`split_by_question`), so paraphrases
   cannot leak across splits.
3. Thresholds and calibration come from validation only; `train_scorer` never
   touches the test split.
4. Your clean corpus must also receive late arrivals from new sources, otherwise
   "new source" means "poison" by construction and S4/S5 score for free.
