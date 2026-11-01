from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
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


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=check,
                          capture_output=True, text=True)


def test_notebook_pin_contains_the_test_suite_implementation():
    """The pin must be an ancestor that already contains the tested code.

    Commit B only updates the pin and documentation, so every implementation,
    test and configuration file must be byte-identical between the pin and
    HEAD.  A stale pin would make Colab run different code from the suite.
    """
    import pytest

    if not (ROOT / ".git").exists():
        pytest.skip("git checkout required to prove the notebook pin")
    notebook = json.loads((ROOT / "notebooks/trace_rag_colab.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    pin = re.search(r"PINNED_COMMIT = \"([0-9a-f]{40})\"", source).group(1)
    head = _git("rev-parse", "HEAD").stdout.strip()
    assert pin != head, "the pin must point at the implementation commit, not the pin commit"
    ancestor = _git("merge-base", "--is-ancestor", pin, head, check=False)
    assert ancestor.returncode == 0, f"notebook pin {pin} is not an ancestor of HEAD {head}"

    changed = _git("diff", "--name-only", pin, head).stdout.splitlines()
    implementation = [path for path in changed
                      if path.startswith(("src/", "scripts/", "tests/", "config/", "constraints/"))
                      or path == "pyproject.toml"]
    assert implementation == [], (
        "notebook pin is stale; implementation/test files differ from HEAD: "
        + ", ".join(implementation))

    preflight_at_pin = _git("show", f"{pin}:scripts/check_research_environment.py").stdout
    assert "PREFLIGHT_STORAGE_SCHEMA" in preflight_at_pin
    assert "isolate_storage_config" in preflight_at_pin
    assert "--preflight-storage-root" in preflight_at_pin
    subset_at_pin = _git("show", f"{pin}:src/trace_rag/ingestion/subset.py").stdout
    assert "_atomic_write" in subset_at_pin
    snapshots_at_pin = _git("show", f"{pin}:src/trace_rag/snapshots.py").stdout
    assert "expected_signature" in snapshots_at_pin


def test_runner_owns_canonical_signature_checked_restore():
    """Recovery must use the runner's one signature path, not a notebook copy."""
    runner = (ROOT / "scripts" / "run_trust_stream.py").read_text(encoding="utf-8")
    assert "expected_signature=signature" in runner
    signature_at = runner.index("signature = run_signature(")
    reuse_at = runner.index("completed_run_matches_signature(out_dir, signature)")
    restore_at = runner.index("restore_latest_snapshot(")
    state_at = runner.index("check_output_state(out_dir, resume)")
    assert signature_at < reuse_at < restore_at < state_at, (
        "recovery must compute the canonical signature, check completed-run reuse, "
        "restore, then validate the restored active directory")
    assert "load_existing_index=resume" in runner

    notebook = json.loads((ROOT / "notebooks" / "trace_rag_colab.ipynb").read_text(
        encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "--restore" in text
    assert "restore_latest_snapshot(out, backup)" not in text
    assert "run_signature(" not in text, "the notebook must not duplicate signature logic"
    assert "completed run already present" not in text
    assert "not overwriting" not in text


def test_notebook_isolates_preflight_storage_and_requires_data_opt_in():
    notebook = json.loads((ROOT / "notebooks" / "trace_rag_colab.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_DATASET_DOWNLOAD = False" in text
    assert "Run dataset download" in text or "Set RUN_DATASET_DOWNLOAD=True" in text
    assert "--preflight-storage-root" in text
    assert "expected_storage_root" in text
    assert "PREFLIGHT_PARENT" in text
    assert "contaminated a fresh experiment directory" in text


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


def test_run_signature_covers_config_model_dataset_seed_and_scope():
    config = Config.load(ROOT / "config" / "colab_pilot_qwen.yaml")
    dataset = {"corpus": {"sha256": "a"}, "queries": {"sha256": "b"}, "qrels": {"sha256": "c"}}
    base = stream.run_signature(config, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                dataset, ["q1"])
    same = stream.run_signature(config, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                dataset, ["q1"])
    assert base == same

    changed_config = config.model_copy(deep=True)
    changed_config.retrieval.top_k = 9
    assert stream.run_signature(changed_config, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                dataset, ["q1"]) != base

    changed_model = config.model_copy(deep=True)
    changed_model.generation.revision = "different"
    assert stream.run_signature(changed_model, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                dataset, ["q1"]) != base

    changed_dataset = json.loads(json.dumps(dataset))
    changed_dataset["corpus"]["sha256"] = "different"
    assert stream.run_signature(config, "pilot", "zipf", 1, 12, 2, 5000, 2000,
                                changed_dataset, ["q1"]) != base
    assert stream.run_signature(config, "pilot", "zipf", 2, 12, 2, 5000, 2000,
                                dataset, ["q1"]) != base
    assert stream.run_signature(config, "pilot", "zipf", 1, 13, 2, 5000, 2000,
                                dataset, ["q1"]) != base


def test_completed_run_reuse_requires_exact_canonical_signature(tmp_path):
    import pytest

    assert stream.completed_run_matches_signature(tmp_path, "sig") is False
    (tmp_path / "metrics.json").write_text(json.dumps({"steps": 4}))
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"state": "completed", "signature": "sig"}))
    (tmp_path / "checkpoint.json").write_text(json.dumps(
        {"signature": "sig", "completed_steps": 4, "in_flight_step": None}))
    assert stream.completed_run_matches_signature(tmp_path, "sig") is True

    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"state": "completed", "signature": "other"}))
    with pytest.raises(ValueError, match="different experiment"):
        stream.completed_run_matches_signature(tmp_path, "sig")

    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"state": "running", "signature": "sig"}))
    with pytest.raises(ValueError, match="not marked completed"):
        stream.completed_run_matches_signature(tmp_path, "sig")

    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"state": "completed", "signature": "sig"}))
    (tmp_path / "checkpoint.json").write_text(json.dumps(
        {"signature": "other", "completed_steps": 4, "in_flight_step": None}))
    with pytest.raises(ValueError, match="checkpoint"):
        stream.completed_run_matches_signature(tmp_path, "sig")

    (tmp_path / "checkpoint.json").unlink()
    with pytest.raises(ValueError, match="missing checkpoint"):
        stream.completed_run_matches_signature(tmp_path, "sig")

    (tmp_path / "run_manifest.json").unlink()
    with pytest.raises(ValueError, match="without run_manifest"):
        stream.completed_run_matches_signature(tmp_path, "sig")


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


def test_restore_flag_requires_a_backup_directory(capsys):
    assert stream.main(["--corpus", "missing.parquet", "--queries", "missing.jsonl",
                        "--restore"]) == 2
    assert "--restore requires --backup-dir" in capsys.readouterr().err


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


def test_exact_query_selection_is_deterministic_and_exact():
    from trace_rag.ingestion.subset import select_query_ids

    gold = {str(i): {f"d{i}"} for i in range(25)}
    first = select_query_ids(gold, 20, seed=7)
    second = select_query_ids(gold, 20, seed=7)
    assert first == second and len(first) == 20 and len(set(first)) == 20
    assert set(first) <= set(gold)


def test_partial_ingestion_without_markers_is_rejected(tmp_path):
    import pytest

    (tmp_path / "provenance.sqlite3").write_bytes(b"partial")
    with pytest.raises(ValueError, match="partial artifacts"):
        stream.check_output_state(tmp_path, resume=False)
    with pytest.raises(ValueError, match="partial artifacts"):
        stream.check_output_state(tmp_path, resume=True)


def test_snapshot_restore_preserves_checkpoint_and_sqlite_state(tmp_path):
    import sqlite3
    from trace_rag.snapshots import backup_snapshot, restore_latest_snapshot

    run, backup = tmp_path / "active", tmp_path / "drive"
    run.mkdir()
    db = run / "trust.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE state (value INTEGER)")
    conn.execute("INSERT INTO state VALUES (1)")
    conn.commit(); conn.close()
    (run / "index.npz").write_bytes(b"index")
    (run / "run_manifest.json").write_text(json.dumps({"signature": "sig"}))
    (run / "stream.jsonl").write_text(json.dumps({"step": 0}) + "\n")
    checkpoint = {"completed_steps": 1, "in_flight_step": None, "signature": "sig"}
    (run / "checkpoint.json").write_text(json.dumps(checkpoint))
    meta = backup_snapshot(run, backup, 1, [db], str(run / "index"), copy_index=True)
    assert meta["completed_steps"] == 1

    conn = sqlite3.connect(db)
    conn.execute("UPDATE state SET value = 2")
    conn.commit(); conn.close()
    (run / "checkpoint.json").write_text(json.dumps(
        {"completed_steps": 0, "in_flight_step": 0, "signature": "sig"}))
    restored = restore_latest_snapshot(run, backup, expected_signature="sig")
    assert restored and restored["completed_steps"] == 1
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT value FROM state").fetchone()[0] == 1
    conn.close()
    assert json.loads((run / "checkpoint.json").read_text())["completed_steps"] == 1


def test_ollama_served_digest_must_match_pinned_revision(monkeypatch):
    import pytest
    from trace_rag.snapshots import snapshot_summary  # noqa: F401

    config = Config.load(ROOT / "config" / "colab_pilot_qwen.yaml")
    config.generation.revision = "sha256:" + "a" * 64
    config.generation.serving_version = "0.5.7"

    def fake(url, payload=None, timeout=5.0):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen2.5:7b", "digest": "sha256:" + "a" * 64}]}
        if url.endswith("/api/show"):
            return {"details": {"quantization_level": "Q4_K_M"}, "capabilities": ["completion"]}
        if url.endswith("/api/version"):
            return {"version": "0.5.7"}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen2.5:7b", "digest": "sha256:" + "a" * 64,
                                "size_vram": 1024}]}
        raise AssertionError(url)

    monkeypatch.setattr(stream, "_http_json", fake)
    runtime = stream.verify_ollama_runtime(config)
    assert runtime["digest_matches_config"] is True
    assert runtime["gpu_resident"] is True

    def wrong(url, payload=None, timeout=5.0):
        result = fake(url, payload, timeout)
        if url.endswith("/api/ps"):
            result = {"models": [{"name": "qwen2.5:7b", "digest": "sha256:" + "b" * 64,
                                  "size_vram": 1024}]}
        return result

    monkeypatch.setattr(stream, "_http_json", wrong)
    with pytest.raises(RuntimeError, match="provenance"):
        stream.verify_ollama_runtime(config)


def test_snapshot_summary_is_read_only(tmp_path):
    from trace_rag.snapshots import snapshot_summary

    assert snapshot_summary(tmp_path / "missing") == {}
    assert snapshot_summary(tmp_path) == {}


def test_notebook_preserves_layout_and_uses_content_for_active_runs():
    notebook = json.loads((ROOT / "notebooks" / "trace_rag_colab.ipynb").read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 30
    assert sum(cell["cell_type"] == "code" for cell in notebook["cells"]) == 15
    headings = [line for cell in notebook["cells"] for line in cell["source"]
                if line.startswith("## ") and line[3:4].isdigit()]
    assert len(headings) == 13
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "/content/ragledger_active" in text
    assert "backup-dir" in text and "--restore" in text
    assert "restore_latest_snapshot(out, backup)" not in text
    assert "constraints/colab.txt" in text
    assert "5,000" in text and "200,000" in text


def test_colab_dependency_pins_are_explicit_and_marked_unverified():
    constraints = (ROOT / "constraints" / "colab.txt").read_text(encoding="utf-8")
    doc = (ROOT / "docs" / "COLAB_DEPENDENCIES.md").read_text(encoding="utf-8")
    for required in ("torch==2.5.1", "transformers==4.48.3", "faiss-cpu==1.9.0",
                     "pyarrow==17.0.0", "vllm==0.7.3"):
        assert required in constraints
    assert "not verified" in doc.lower() and "0.5.7" in doc
    runbook = (ROOT / "docs" / "COLAB_RUNBOOK.md").read_text(encoding="utf-8")
    assert "backups" in runbook
