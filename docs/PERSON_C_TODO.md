# Person C evaluation status (updated 2026-10)

This file replaces the historical audit at commit `9a2f88f`. That audit found
multiple disconnected/scaffold-only paths; those defects have since been fixed
and regression-tested. Do not use its original zero-coverage counts or bug
claims as current status.

## Implemented and regression-tested

- Typed attack and stream events serialize to JSON-friendly dictionaries.
- Stream event/query adapters match the chronological runner's input schema.
- The stream runner enforces that a poison is ingested before its target query.
- Attack transformations include targeted entity replacement, contradictory
  negation, slow-burn warm-up documents, and signal-evasion payloads.
- Baseline policies and metrics are present; undefined denominators return
  `None`, not an optimistic zero.
- `EvaluationCase`/`EvaluationResult`, deterministic matrix construction and a
  callback-based matrix executor are available.
- Bootstrap confidence intervals and exact McNemar tests are available.
- Unit/integration regression coverage is in `tests/test_person_c.py` and
  related attack, stream, metrics, and baseline tests.

## Still not a complete research-results harness

The callback-based matrix executor does not itself implement and validate every
attack × baseline × seed case against a real benchmark. In particular:

1. There is no full, independently reviewed benchmark protocol that maps every
   advertised baseline to the exact same chronological inputs and records all
   attack, false-positive, exposure, remediation, cost, and citation metrics.
2. The bundled mini-corpus smoke remains only a plumbing/security regression.
   The Colab TPU notebook now contains a full-BEIR-NQ/Ollama integration path and
   a one-query real-data baseline API diagnostic, but neither has been executed
   in this development environment. They are not evidence of performance on
   Natural Questions or another benchmark until a live run is completed and
   audited.
3. Real embedding, verifier, and answer-generation weights have not all been
   run end to end here. A run must record the actual `verifier_mode`, model
   identifiers, full-corpus/query-sample scope, configuration, random seed, and
   code revision. BEIR NQ lacks original source identities and timestamps, and
   controlled poisoning examples are generated mutations rather than real
   incidents.
4. Reported result tables, confidence intervals, and statistical comparisons
   must be generated from saved per-run artifacts; no paper headline number is
   established by the unit tests or a mini smoke.

Before publishing, implement/execute the complete benchmark matrix on the
selected dataset, validate baseline parity and labels, retain raw per-query
artifacts, and report confidence intervals across independent seeds. See
`docs/REQUIREMENTS.md` and `docs/PRODUCTION_READINESS.md` for the broader
limitations.
