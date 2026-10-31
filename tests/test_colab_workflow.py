from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

from trace_rag.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_script("check_research_environment")
stream = _load_script("run_trust_stream")


def _questions(n: int):
    return [{"qid": str(i), "question": f"question {i}", "gold_answer": "answer"}
            for i in range(n)]


def test_notebook_is_valid_and_every_code_cell_parses():
    notebook = json.loads((ROOT / "notebooks" / "trace_rag_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) >= 15
    for cell in code:
        assert cell.get("outputs") == []
        ast.parse("".join(cell["source"]))


def test_notebook_uses_the_fork_and_an_explicit_commit():
    source = "".join("".join(cell["source"])
                     for cell in json.loads((ROOT / "notebooks/trace_rag_colab.ipynb").read_text(
                         encoding="utf-8"))["cells"])
    assert "athishss/-Adaptive-Trust-Based-Framework" in source
    assert "vaishnavissl1" not in source
    match = re.search(r"PINNED_COMMIT = \"([0-9a-f]{40})\"", source)
    assert match, "notebook must expose a full 40-character pinned SHA"
    assert '"rev-parse", "HEAD"' in source and "checked-out SHA" in source


def test_notebook_has_the_required_linear_sections_and_persistence():
    notebook = json.loads((ROOT / "notebooks/trace_rag_colab.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    markers = [
        "GPU verification", "Clone the correct repository", "Mount Google Drive",
        "Install dependencies", "Cache the embedding and NLI weights",
        "Download raw NQ data", "Functional preflight", "Integration pilot",
        "Separate research execution", "Resume after a runtime disconnection",
        "Metrics, manifest and visualization", "Export and preserve",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "MyDrive/RAGLedger" in text and "HF_HOME" in text
    assert "check_research_environment.py" in text and "--functional" in text
    assert "--run-purpose\", \"pilot" in text or '"--run-purpose", purpose' in text
    assert "--resume" in text
    assert "distinct_queries_evaluated" in text
    assert "NOT MEASURED" in text


def test_notebook_never_claims_pilot_model_is_the_research_model():
    text = (ROOT / "notebooks" / "trace_rag_colab.ipynb").read_text(encoding="utf-8")
    assert "qwen2.5:7b" in text
    assert "meta-llama/Llama-3.1-8B-Instruct" in text
    assert "equivalence_claim" in text
    assert "not equivalent" in text.lower() or "never equivalent" in text.lower()


def test_pilot_and_research_configs_are_explicit_and_share_science_settings():
    pilot = Config.load(ROOT / "config" / "colab_pilot_qwen.yaml")
    research = Config.load(ROOT / "config" / "nq_gpu.yaml")
    assert pilot.embedding.model_name == research.embedding.model_name == "facebook/contriever"
    assert pilot.embedding.revision == research.embedding.revision
    assert pilot.security.nli_model == research.security.nli_model
    assert pilot.security.nli_revision == research.security.nli_revision
    assert pilot.trust.model_dump() == research.trust.model_dump()
    assert pilot.generation.model_name == "qwen2.5:7b"
    assert pilot.generation.backend == "ollama"
    assert research.generation.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert research.generation.backend == "openai"
    assert pilot.generation.model_name != research.generation.model_name
    assert pilot.generation.quantization and research.generation.precision
    assert pilot.storage.root != research.storage.root


def test_preflight_resolves_the_same_runner_overrides():
    config = preflight.resolve_config(
        ROOT / "config" / "nq_gpu.yaml",
        ["generation.backend=ollama", "generation.model_name=qwen2.5:7b",
         "generation.revision=sha256:test", "embedding.revision=abc"],
    )
    assert config.generation.backend == "ollama"
    assert config.generation.model_name == "qwen2.5:7b"
    assert config.generation.revision == "sha256:test"
    assert config.embedding.revision == "abc"


def test_query_stream_separates_steps_from_distinct_queries():
    questions = _questions(20)
    repeated = stream.build_query_stream(questions, 12, seed=7, sample_mode="zipf")
    unique = stream.build_query_stream(questions, 12, seed=7, sample_mode="no-repeat")
    assert len(repeated) == 12
    assert len(set(repeated)) <= 12
    assert len(unique) == 12 and len(set(unique)) == 12


def test_no_repeat_refuses_to_shorten_a_request():
    import pytest

    with pytest.raises(ValueError, match="only"):
        stream.build_query_stream(_questions(3), 5, seed=1, sample_mode="no-repeat")


def test_scope_validation_rejects_mislabeled_pilot_and_research():
    assert stream.validate_run_scope("pilot", True, 5000, 12, 2, 20, "zipf") == []
    assert stream.validate_run_scope("research", True, None, 500, 5, 500, "no-repeat") == []
    assert stream.validate_run_scope("research", True, None, 40, 5, 500, "zipf")
    assert stream.validate_run_scope("pilot", True, None, 12, 2, 20, "zipf")
    assert stream.validate_run_scope("smoke", True, None, 10, 1, 10, "zipf")


def test_run_signature_changes_with_model_revision():
    config = Config.load(ROOT / "config" / "colab_pilot_qwen.yaml")
    dataset = {"corpus": {"sha256": "a"}, "queries": {"sha256": "b"}, "qrels": {"sha256": "c"}}
    base = stream.run_signature(config, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                dataset, ["q1"])
    same = stream.run_signature(config, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                dataset, ["q1"])
    assert base == same
    changed = config.model_copy(deep=True)
    changed.generation.revision = "different"
    other = stream.run_signature(changed, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                 dataset, ["q1"])
    assert other != base


def test_output_guard_refuses_overwrite_and_mixed_resume(tmp_path):
    import pytest

    assert stream.check_output_state(tmp_path, resume=False) is None
    (tmp_path / "checkpoint.json").write_text(json.dumps({"completed_steps": 2,
                                                          "in_flight_step": None}))
    (tmp_path / "run_manifest.json").write_text(json.dumps({"signature": "x"}))
    with pytest.raises(ValueError, match="already contains"):
        stream.check_output_state(tmp_path, resume=False)
    assert stream.check_output_state(tmp_path, resume=True)["completed_steps"] == 2
    (tmp_path / "checkpoint.json").write_text(json.dumps({"completed_steps": 2,
                                                          "in_flight_step": 2}))
    with pytest.raises(ValueError, match="inside a stream step"):
        stream.check_output_state(tmp_path, resume=True)


def test_runbook_documents_the_fork_drive_preflight_and_both_scopes():
    text = (ROOT / "docs" / "COLAB_RUNBOOK.md").read_text(encoding="utf-8")
    assert "athishss/-Adaptive-Trust-Based-Framework" in text
    assert "MyDrive/RAGLedger" in text
    assert "check_research_environment.py" in text
    assert "--run-purpose pilot" in text
    assert "--run-purpose research" in text
    assert "5,000" in text and "200,000" in text and "500" in text
    assert "NOT MEASURED" in text
    assert "faiss-cpu" in text
