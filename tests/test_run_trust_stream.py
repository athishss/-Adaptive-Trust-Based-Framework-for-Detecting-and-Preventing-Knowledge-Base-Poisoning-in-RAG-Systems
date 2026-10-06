from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest


_script = run_path("scripts/run_trust_stream.py")
corrupt_passage = _script["corrupt_passage"]
load_questions = _script["load_questions"]


def test_load_questions_keeps_type_compatible_false_answers(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({
        "questions": [{"qid": "everest", "question": "How high is Mount Everest?",
                       "gold_answer": "8849 metres"}],
        "false_answers": {"everest": "11200 metres"},
    }), encoding="utf-8")
    question = load_questions(path)[0]
    assert question["false_answer"] == "11200 metres"


def test_numeric_poison_uses_a_numeric_false_answer():
    poison, attack = corrupt_passage(
        "Mount Everest has a summit elevation of 8849 metres.",
        "8849 metres", "11200 metres",
    )
    assert attack == "poisonedrag_bbox"
    assert "11200 metres" in poison
    assert "8849 metres" not in poison


def test_numeric_fallback_preserves_answer_type():
    poison, attack = corrupt_passage("The result is 100 metres.", "100 metres")
    assert attack == "poisonedrag_bbox"
    assert "metres" in poison
    assert "Zog the Alien" not in poison


def test_mini_stream_writes_reproducible_metrics_without_clean_quarantine(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "mini"
    result = subprocess.run(
        [
            sys.executable, "scripts/run_trust_stream.py",
            "--config", "config/default.yaml",
            "--corpus", "examples/mini_corpus.jsonl",
            "--queries", "examples/mini_questions.json",
            "--steps", "12", "--targets", "2", "--out", str(out),
            "--save-every", "5",
            "--set", "trust.nli_mode=lexical",
            "--set", "generation.backend=stub",
        ],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["verifier_mode"] == "lexical"
    assert metrics["poison_quarantined"] == 2
    assert metrics["clean_false_quarantine"] == 0
    assert metrics["poison_citation_rate"] == 0.0
    assert metrics["embedding_device"] == "cpu"
    assert metrics["generation_backend"] == "stub"
    assert metrics["generation_model"] == "stub"
    assert metrics["generation_device"] == "not-applicable"
    assert metrics["clean_source_mode"] == "simulated_contributors"
    assert metrics["clean_source_id"] is None
    assert metrics["live_timestamps"] is False
    assert metrics["provenance_note"].startswith("Clean contributor/source identities are simulated.")
    assert (out / "resolved_config.yaml").is_file()
    from trace_rag.config import Config
    assert Config.load(out / "resolved_config.yaml").trust.enabled is True
    assert (out / "stream.jsonl").is_file()
    assert (out / "trust_history.csv").is_file()


def test_title_source_live_timestamp_mode_is_recorded(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "title-source-run"
    result = subprocess.run(
        [
            sys.executable, "scripts/run_trust_stream.py",
            "--config", "config/default.yaml",
            "--corpus", "examples/mini_corpus.jsonl",
            "--queries", "examples/mini_questions.json",
            "--steps", "4", "--targets", "1", "--out", str(out),
            "--source-from-title", "--live-timestamps", "--save-every", "0",
            "--set", "trust.nli_mode=lexical",
            "--set", "generation.backend=stub",
        ],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["clean_source_mode"] == "title_based"
    assert metrics["source_title_prefix"] == "wiki-page"
    assert metrics["live_timestamps"] is True
    assert "record titles at page level" in metrics["provenance_note"]
