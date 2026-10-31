# Colab dependency pins (target, not verified)

`constraints/colab.txt` records the intended Python 3.11 / CUDA stack for the
Colab pilot and Llama research path. It is a **target lock, not a compatibility
claim**: no Colab runtime has been executed in this repository. The notebook
installs with `-c constraints/colab.txt` and records the versions actually
imported; it fails rather than silently accepting a different major stack.

| Component | Pin |
|---|---|
| PyTorch | 2.5.1 |
| Transformers | 4.48.3 |
| huggingface-hub | 0.27.1 |
| FAISS CPU | 1.9.0 |
| PyArrow | 17.0.0 |
| NumPy / SciPy | 1.26.4 / 1.14.1 |
| scikit-learn | 1.5.2 |
| vLLM (research server) | 0.7.3 |
| Ollama (pilot server) | 0.5.7 |

Ollama's model tag is mutable. The notebook resolves the pulled manifest digest,
writes it into the resolved config and model lock, and the runner refuses a run
unless `/api/ps` reports the same digest and `/api/version` matches the pinned
server version. The Qwen digest itself is recorded at pull time, never guessed.

The versions above are only a starting compatibility set. They must be tested
in the selected Colab image before a real experiment; the notebook records the
observed versions in `run_manifest.json` and the preflight JSON.

Local repository validation installed `pyarrow 26.0.0` only to exercise the
Parquet subset code; that is **not** evidence that the `pyarrow==17.0.0` Colab
pin works. `torch`, `transformers` and `faiss` were not installed on the local
validation host, so their tests were skipped.
