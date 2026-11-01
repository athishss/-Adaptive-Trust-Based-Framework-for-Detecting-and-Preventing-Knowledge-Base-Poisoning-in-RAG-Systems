# RAGLedger validation evidence

## Current stabilization pass

- Tested validation tree: `1760828b1e90419c7dd91f4ee691344c24fa5f1b`
- Implementation commit pinned by the notebook: `32333fe125c446d043bd422e9bbd86005d1259d8`
- Environment: Windows 11 (10.0.26200), CPython 3.13.3, no GPU
- Optional backends absent: `faiss-cpu`, `torch`, `matplotlib`

| Check | Result | Log |
|---|---|---|
| Full suite | **347 passed, 6 skipped**, exit 0 | `stabilization_pytest.txt` |
| Coverage | **81%**, 6319 statements, 1214 missed | `stabilization_coverage.txt` |
| ResourceWarning as error | **347 passed, 6 skipped**, exit 0, no ResourceWarning | `stabilization_warnings.txt` |
| Compilation | `compileall -q src scripts tests`, exit 0 | `stabilization_compileall.txt` |
| Focused workflow tests | 90 passed, exit 0 | `stabilization_focused.txt` |
| Notebook JSON/AST/pin/layout | 24 passed, exit 0 | `stabilization_notebook.txt` |
| Environment | Python 3.13.3, Windows 11, package freeze | `stabilization_environment.txt` |
| Offline mini stream | run 0, exact reuse 0, changed-scope rejection 2, restore 0 | `stabilization_mini_stream.txt` |

The mini stream executed 4/4 steps, recorded a step-4 snapshot with the prepared
index, reused the matching completed run with zero executed steps, rejected a
`--steps 3` rerun as a different experiment, and restored a completed run after
the active directory was moved without re-executing a step. All four
`metrics.json` SHA-256 hashes were identical.

## Skips

The six skips are environment-specific and are not passes:

- `faiss-cpu` absent: 2 index/retrieval tests.
- `torch` absent: 1 model-backend test.
- `matplotlib` absent: 3 trust-plot tests.

## Caveats

- No Google Colab GPU execution occurred. The pilot and research trust-stream
  remain unexecuted, and answer-level ASR remains **NOT MEASURED**.
- The offline mini stream used `config/default.yaml`, hashing embeddings, the
  numpy index and the lexical verifier. It is not evidence for Contriever,
  FAISS, NLI, Qwen or Llama.
- Coverage is from the exact log above. Scripts are not part of the
  `--cov=trace_rag` report.

## Earlier snapshot-recovery pass

The `colab_reliability_*` logs are historical evidence for validation tree
`25b582c10fab1b08d2dcd1e4ea082436888bc93a` and implementation
`8bf874a10ea2f95b30f8b3e7de15aacc99e7baa0`. They are superseded by this pass.
