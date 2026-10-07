# Full BEIR NQ run on a Colab T4

Notebook: [`notebooks/trace_rag_colab_t4.ipynb`](../notebooks/trace_rag_colab_t4.ipynb)

This is a full-passage-corpus integration run, not a small-corpus smoke. It targets an NVIDIA T4 CUDA runtime, real Hugging Face Contriever and NLI models, a local real Ollama Qwen2.5 3B generator, and the repository's Person A/B/C pipeline. The notebook checks for a visible GPU whose reported name contains `T4` and fails early otherwise.

## Runtime and resources

1. Open the notebook in Colab and select **Runtime → Change runtime type → T4 GPU**. Restart, then run all cells from the top.
2. The notebook preserves Colab's installed PyTorch/CUDA build. It installs the project, `faiss-cpu`, Parquet/plotting/test dependencies, and Transformers; it does not request a PyTorch replacement.
3. The checked-in `config/nq_t4.yaml` uses `facebook/contriever` on CUDA in float16, CUDA NLI verification, and CPU FAISS IVF-PQ. Embedding batch size is 8 when at least 14 GiB of GPU memory is detected and 4 otherwise. NLI sequence length is 256. Ollama is configured for one parallel request and one loaded model to limit T4 memory contention.
4. Ollama downloads and runs the real `qwen2.5:3b` model. The notebook records `ollama ps`; use that output to check whether Ollama offloaded to the T4 or is using CPU. The embedding and verifier device metrics must independently report CUDA.
5. The complete BEIR NQ corpus is downloaded and indexed (expected 2,681,468 passages); the notebook samples 500 authentic qrels-backed queries for the stream. Expect multi-GB downloads, substantial storage/RAM usage, and a potentially multi-hour indexing run. It does not create a smaller corpus subset.

The stream checkpoints the FAISS index every 50,000 passages. To resume an interrupted indexing run in the same Colab VM while its files are still available, set `RESUME_RUN_ID` in the test/run setup cell to the earlier UTC run-folder name and set `RESUME_INDEX = True` in the trust-stream cell. Do not use this for a completed run or to resume a partially completed attack/query stream; it resumes the corpus index, not the evaluation protocol. If the T4 reports less memory or a CUDA OOM occurs, restart and lower `NQ_EMBED_BATCH_SIZE` to 4; keep Ollama's single-model settings.

## Data and evaluation limits

The corpus, queries, and qrels are authentic BEIR Natural Questions. The full passage corpus is indexed, but only the deterministic 500-query sample drives this notebook; its one-query baseline diagnostic is not a matched head-to-head benchmark. BEIR NQ does not contain contributor identities, original ingestion timestamps, or real poisoning incident data. The run derives page-level source IDs from real titles, timestamps imports at runtime, and generates controlled poison mutations from authentic qrels-relevant passages. These generated attack cases are experimental inputs—not authentic provenance or real incidents. A literal no-generated-attack study requires real source metadata and incident documents.

The notebook exercises implemented retrieval/generation, trust/verification/remediation, attacks, and baseline/evaluation code paths, but does not produce a complete attack × baseline × seed benchmark or establish production readiness. No full-corpus T4/Ollama execution is claimed from this development environment; validate the run on live Colab hardware before relying on its outputs.
