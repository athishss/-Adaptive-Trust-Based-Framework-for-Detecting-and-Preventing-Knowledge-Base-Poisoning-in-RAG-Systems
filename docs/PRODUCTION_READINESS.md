# Production-readiness assessment (updated 2026-10)

## Decision

**TRACE-RAG is not production-ready for deployment in a company.** It is a research prototype and CLI/library with an end-to-end demonstration path. A passing test suite, the bundled mini-corpus smoke, or a successful Colab run does not certify security, safety, accuracy, scale, or availability for an organization's workload.

## What has been strengthened in this pass

- The reproducible mini stream now detects/quarantines both injected poisons without quarantining clean documents in the tested CPU + lexical-verifier + stub-LLM fixture. The regression is pinned in `tests/test_run_trust_stream.py`.
- Numeric counterfactual poisons preserve answer type; refutation no longer matures a source's cold-start influence.
- The lexical refutation gate checks the question's predicate family as well as topic, reducing same-subject/wrong-relation false refutations.
- Explicit `trust.nli_mode: nli` now fails fast if the verifier model/device is unavailable rather than silently changing the experiment to lexical mode. `auto` remains an offline-development convenience; always record the actual verifier mode.
- NLI input is passed as a premise/hypothesis pair. TPU/XLA paths exist for embeddings, NLI, and optional local Hugging Face generation, with static XLA embedding batch/sequence shapes and a TPU config.
- Stream artifacts include validated resolved configuration and actual backend/device metadata. The optional Colab notebook can save raw run files, a CSV summary, test output, and a ZIP archive.
- S6 uses the target document ID rather than assuming the target is the first nearest-neighbour result.

The mini fixture is intentionally not representative evidence. Its current measured CPU smoke is recorded only as a regression: 2 injected documents quarantined, 0 clean false quarantines, 0 poison citations, and 0.5 poison retrieval rate; the verifier and answer backend are lexical/stub. This must not be presented as a benchmark result.

## Evidence and unverified items

- The repository test suite passed in the development sandbox. Optional tests were skipped because FAISS, torch/Transformers, and Matplotlib are not installed there.
- No live TPU, CUDA GPU, Hugging Face model weights, full NQ run, or real answer model was available in the development sandbox. The TPU implementation is **not hardware-validated** until the TPU notebook completes in Colab and its actual device metadata shows XLA.
- `scripts/run_trust_stream.py` is a component smoke runner. It is not the complete controlled attack × baseline × seed research harness with confidence intervals. The generic callback matrix is not itself benchmark execution.
- The codebase is not a hosted, multi-tenant service. It does not provide organization-specific identity, authorization, network isolation, tenant quotas, key management, or a deployment control plane.

## Company deployment blockers

Before any company use, an owner must design, implement, and verify at least:

1. **Threat model and security review:** define attacker capabilities, poisoning/prompt-injection scenarios, trust-recovery and admin-review policy, fail-open/fail-closed behavior, abuse cases, and a red-team test set. Lexical/NLI checks are heuristics, not a security boundary.
2. **Identity and tenant isolation:** authentication, authorization/RBAC, tenant-scoped provenance/trust/index storage, isolation tests, quotas, request validation, and safe administrative access to quarantine/review queues.
3. **Data protection:** data classification, PII handling, retention/deletion, encryption in transit/at rest, backup/restore, audit-log integrity, and approved model/data egress. These depend on the target organization's environment and policy.
4. **Operational reliability:** service/API integration, concurrency/load testing at target corpus size, resource limits, backpressure, restart/recovery, monitoring/alerts, SLOs, incident runbooks, and rollback procedures.
5. **Model and dataset validation:** fixed/pinned model revisions, provenance for training/evaluation data, human-adjudicated labels, representative clean/adversarial workloads, multiple seeds, baseline parity, confidence intervals, and measured quality/latency/cost trade-offs.
6. **Supply-chain controls:** a deployment lockfile/constraints for the exact Python, torch, torch_xla, Transformers, FAISS and model revisions; license review; vulnerability scanning; artifact integrity; and a documented update/rollback process. The broad version ranges in `pyproject.toml` are not a production lockfile.
7. **Human review and governance:** define when quarantine/remediation requires approval, how false positives are appealed, how trust changes are explained, and how model/policy updates are approved.

## Recommended current use

Use this repository for research iteration, tests, controlled demos, and collecting **clearly labelled exploratory** Colab artifacts. For research results, complete the benchmark protocol and record code revision, resolved config, actual accelerator/verifier device, embedding and generation model revisions, dataset/subset hashes, all seeds, per-query outputs, and statistical uncertainty. For company deployment, complete the blockers above and get the target organization's security, privacy, and platform review before serving real data or users.
