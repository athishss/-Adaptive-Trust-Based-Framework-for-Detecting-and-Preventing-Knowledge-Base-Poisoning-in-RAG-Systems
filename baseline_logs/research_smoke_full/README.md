# Offline fixture evidence

- `run.log`: full 12-attack × 11-policy run, 132/132 cases, no failures.
- `analysis.{json,csv,md}`: per-attack/policy aggregates from that run.
- `multiseed_*.log`: three independent seeds at two targets per case.

Scope: bundled mini corpus, hashing embedder, stub generator, lexical verifier.
This is **not** Natural Questions, BEIR, or any research benchmark. The
single-seed analysis has detection precision/recall 0 in every group; the
three-seed run has isolated non-zero values (0.333/0.167 for adaptive and
entity-swap) but remains weak. Do not claim detector efficacy. `analysis.md` is
the compact table used by the manuscript; `multiseed_analysis.*` is the
variance check.
