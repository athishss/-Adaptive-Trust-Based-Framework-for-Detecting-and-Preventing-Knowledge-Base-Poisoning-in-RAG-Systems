# Related-work positioning (verified sources)

This note records the papers used to position TRACE-RAG. It is not a claim that
the prototype reproduces or outperforms them. All links were checked while
building the manuscript draft.

| Work | Venue / ID | Core idea | Difference from this prototype |
|---|---|---|---|
| PoisonedRAG | arXiv:2402.07867 | Knowledge-corruption attacks that inject a small number of adversarial passages to steer RAG answers. | TRACE-RAG studies defence with chronological ingestion, provenance and remediation; it does not claim a new attack. |
| TrustRAG | arXiv:2501.00879 | Trust-aware retrieval/response filtering for robust RAG. | TRACE-RAG adds persistent cross-document trust, deferred verification and retroactive answer remediation. The baseline here is a label-free adaptation, not the published algorithm. |
| RobustRAG | arXiv:2405.15556 | Certified robust aggregation of isolated responses. | The baseline here is band blocking, not certified aggregation. TRACE-RAG is provenance-aware rather than isolation-only. |
| RAGForensics | arXiv:2504.21668 | Traces poisoning sources after suspicious RAG output. | TRACE-RAG records provenance continuously and can quarantine and re-open earlier answers; forensic tracing is not the primary mechanism. |
| FilterRAG | arXiv:2508.02835 | Retrieval filtering against knowledge poisoning. | TRACE-RAG combines suspicion/trust with independent corroboration and delayed release, not a single filter. |
| RAGDefender | arXiv:2511.01268 | Detection and mitigation of poisoned knowledge in RAG. | The prototype differs by retaining document/family/source trust and remediating past answers. No claim of superiority is made. |
| SafeRAG | arXiv:2501.18636 | Security benchmark for RAG. | SafeRAG informs evaluation design; this repository's fixture is not SafeRAG and is not a benchmark substitute. |
| CorruptRAG | arXiv:2504.03957 | Large-scale dataset for RAG security evaluation. | The local fixture is tiny and synthetic; CorruptRAG is future evaluation work, not executed here. |
| RAGShield | arXiv:2604.00387 | Recent provenance/defence framework for RAG poisoning (preprint). | The closest provenance-related work. TRACE-RAG's distinctive combination is provenance-independent corroboration, deferred release and retroactive answer remediation/blast-radius accounting. This must be verified against the final RAGShield version before submission. |

## Positioning statement

The defensible gap is **not** "first to detect poisoned RAG documents." Prior
work already covers poisoning attacks, trust filtering, certified aggregation,
forensics and provenance defences. The prototype's narrow contribution is an
integrated, chronological mechanism that:

1. treats ingestion as a stream rather than a static corpus;
2. separates suspicion, source trust and independent corroboration;
3. defers release when evidence is unresolved; and
4. retroactively identifies previously delivered answers that depended on
   content quarantined later.

The last point, with blast-radius accounting, is the strongest differentiator.
The current evidence does not establish that the mechanism works on a real
benchmark, so the manuscript must present it as a design/protocol contribution
plus a negative/plumbing result, not as an efficacy claim.

## Citation hygiene

- Cite the arXiv identifiers above, not invented venue pages.
- Verify the final RAGShield authorship/version immediately before submission;
  it is a recent preprint and may change.
- Do not describe adapted baselines as the published TrustRAG or RobustRAG
  implementations. `BASELINE_SPECS` marks them `published_method=False`.
- Do not cite the offline fixture as Natural Questions, BEIR or a benchmark.
