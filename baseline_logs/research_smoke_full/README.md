# Offline fixture evidence

- `run.log`: full 12-attack × 11-policy run, 132/132 cases, no failures.
- `analysis.{json,csv,md}`: per-attack/policy aggregates from that run.
- `multiseed_*.log`: three independent seeds at two targets per case.

Scope: bundled mini corpus, hashing embedder, stub generator, lexical verifier.
This is **not** Natural Questions, BEIR, or any research benchmark. Detection
precision/recall are 0 in every group; do not claim detector efficacy from this
run. `analysis.md` is the compact table used by the manuscript.
