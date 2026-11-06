"""The real-results notebooks: valid, pinned to tested code, and only call flags that exist."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "notebooks" / "ragledger_real_results.ipynb"
BASE = ROOT / "notebooks" / "ragledger_baselines_official.ipynb"


def _cells(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb["cells"]


def _code(path: Path) -> str:
    return "\n".join("".join(c["source"]) for c in _cells(path) if c["cell_type"] == "code")


@pytest.mark.parametrize("path", [MAIN, BASE])
def test_every_code_cell_parses_and_ships_without_outputs(path):
    for cell in _cells(path):
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            ast.parse("".join(cell["source"]))


def test_main_notebook_uses_real_components_only():
    text = _code(MAIN)
    for banned in ("StubLLM", "backend=stub", "generation.backend=stub", "hashing", "embedding.backend=hashing"):
        assert banned not in text, banned
    assert "poisonedrag_official" in text and "trustrag_official" in text
    assert "RUN_DATASET_DOWNLOAD = False" in text            # multi-GB download stays opt-in
    assert "--include-queries" in text and "--target-offset" in text


def test_main_notebook_pin_is_an_ancestor_with_identical_implementation():
    if not (ROOT / ".git").exists():
        pytest.skip("git checkout required")
    pin = re.search(r'PINNED_COMMIT = "([0-9a-f]{40})"', _code(MAIN)).group(1)
    git = lambda *a, check=True: subprocess.run(["git", "-C", str(ROOT), *a], check=check,
                                                capture_output=True, text=True)
    head = git("rev-parse", "HEAD").stdout.strip()
    assert git("merge-base", "--is-ancestor", pin, head, check=False).returncode == 0
    changed = git("diff", "--name-only", pin, head).stdout.splitlines()
    impl = [p for p in changed if p.startswith(("src/", "scripts/", "tests/", "config/", "constraints/"))
            or p == "pyproject.toml"]
    assert impl == [], "notebook pin is stale; implementation differs from HEAD: " + ", ".join(impl)


def test_every_script_flag_the_notebook_passes_exists():
    text = _code(MAIN)
    helps = {}
    for script in sorted(set(re.findall(r"scripts/([a-z_]+\.py)", text))):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                             capture_output=True, text=True)
        assert out.returncode == 0, script
        helps[script] = out.stdout
    used = {
        "run_poisonedrag_official.py": ["--poison", "--corpus", "--qrels", "--config", "--out", "--targets",
                                         "--max-clean", "--attacks", "--baselines", "--seeds",
                                         "--trustrag-dir", "--scope", "--set"],
        "train_suspicion_scorer.py": ["--config", "--out", "--poisonedrag-json", "--corpus", "--qrels",
                                       "--targets", "--target-offset", "--max-clean", "--attacks", "--set"],
        "analyze_results.py": ["--reference", "--scope"],
        "download_data.py": ["--dataset", "--out", "--subset", "--queries", "--include-queries"],
        "check_research_environment.py": ["--config", "--functional", "--json", "--preflight-storage-root",
                                           "--corpus", "--queries", "--qrels", "--set"],
    }
    assert set(used) == set(helps), set(helps) ^ set(used)
    for script, flags in used.items():
        for flag in flags:
            assert flag in helps[script], f"{script} has no {flag}"
            assert f'"{flag}"' in text, f"notebook never passes {flag}"


def test_baseline_notebook_pins_robustrag_and_says_it_is_not_paired():
    text = "\n".join("".join(c["source"]) for c in _cells(BASE))
    assert re.search(r'SHA = "[0-9a-f]{40}"', text)
    assert "not paired" in text and "run as published" in text
