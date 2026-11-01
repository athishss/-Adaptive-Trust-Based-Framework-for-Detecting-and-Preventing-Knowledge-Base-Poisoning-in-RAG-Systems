# RAGLedger validation evidence

This directory contains the offline validation evidence for the snapshot-safety
and documentation release-readiness pass. It is **not** a Colab GPU run.

- Tested validation tree: `25b582c10fab1b08d2dcd1e4ea082436888bc93a`
- Implementation commit pinned by the notebook: `8bf874a10ea2f95b30f8b3e7de15aacc99e7baa0`
- Environment: Windows 11 (10.0.26200), CPython 3.13.3, no GPU
- Optional backends absent: `faiss-cpu`, `torch`, `matplotlib`

## Results

| Check | Result | Log |
|---|---|---|
| Full suite | **346 passed, 6 skipped**, exit 0 | `colab_reliability_pytest.txt` |
| Coverage | **81%**, 6319 statements, 1214 missed | `colab_reliability_coverage.txt` |
| ResourceWarning as error | **346 passed, 6 skipped**, exit 0, no ResourceWarning | `colab_reliability_warnings.txt` |
| Compilation | `compileall -q src scripts tests`, exit 0 | `colab_reliability_compileall.txt` |
| Focused workflow tests | 89 passed, exit 0 | `colab_reliability_focused.txt` |
| Notebook JSON/AST/pin/layout | 4 passed, exit 0 | `colab_reliability_notebook.txt` |
| Environment | Python 3.13.3, Windows 11, package freeze | `colab_reliability_environment.txt` |
| Offline mini stream and restore | exit 0; 4/4 steps; snapshot at step 4; metrics byte-identical after restore; no steps re-executed | `colab_reliability_mini_stream.txt` |

Focused workflow tests cover `tests/test_colab_workflow.py`,
`tests/test_research_preflight.py`, `tests/test_ingestion_subset.py` and
`tests/test_snapshots.py`. The notebook tests assert 30 cells, 15 code cells,
13 numbered sections, no saved outputs, every code cell parses, and the pin is
an ancestor of HEAD with no implementation/test differences.

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
- Coverage figures are from the exact log above. The subset module is 91% and
  the snapshot module is 86%.
