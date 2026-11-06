"""TrustRAG's own stage-1 filter, run on this pipeline's retrieved passages."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import make_doc as _doc


def _fake_torch(monkeypatch):
    import sys
    import types
    # defend_module imports torch; the stage-1 filter itself does not use it.
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def test_missing_checkout_is_a_clear_error(tmp_path):
    from trace_rag.baselines.trustrag_official import load_defend_module

    with pytest.raises(FileNotFoundError, match="HuichiZhou/TrustRAG"):
        load_defend_module(tmp_path)


def test_trustrag_official_is_opt_in_and_needs_a_checkout():
    from trace_rag.evaluation.experiment import (ALL_BASELINES, OPT_IN_BASELINES,
                                                 SMOKE_BASELINES, load_fixture, build_pipeline)
    assert "trustrag_official" in OPT_IN_BASELINES and "trustrag_official" in ALL_BASELINES
    assert "trustrag_official" not in SMOKE_BASELINES
    fixture = load_fixture(Path(__file__).resolve().parent.parent / "examples")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="trustrag_dir"):
            build_pipeline(None, tmp, "trustrag_official", fixture)


def test_stage1_runs_the_authors_code_when_a_checkout_is_available(monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("nltk")
    pytest.importorskip("rouge_score")
    repo = Path(os.environ.get("TRUSTRAG_DIR", "/tmp/claude-0/ext/TrustRAG"))
    if not (repo / "defend_module.py").exists():
        pytest.skip("no TrustRAG checkout (git clone https://github.com/HuichiZhou/TrustRAG.git)")
    _fake_torch(monkeypatch)
    from trace_rag.baselines.trustrag_official import TrustRAGOfficialStage1Policy

    rng = np.random.default_rng(0)
    clean = [_doc(f"c{i}", f"clean passage {i} about rivers mountains and weather patterns number {i * 7}")
             for i in range(3)]
    poison = [_doc(f"p{i}", "The tower was designed by Zog the Alien in 1889 and everyone agrees.")
              for i in range(2)]
    docs = clean + poison
    base_clean, base_poison = rng.normal(size=16), rng.normal(size=16)

    def embed(texts):
        return np.array([(base_poison if "Zog" in t else base_clean) + rng.normal(scale=0.01, size=16)
                         for t in texts])

    policy = TrustRAGOfficialStage1Policy(repo, embed)
    decision = policy.decide("q", "q1", docs, [], None)
    assert set(decision.context_doc_ids) | set(decision.excluded_doc_ids) == {d.doc_id for d in docs}
    assert decision.notes["stage2_included"] is False
    assert policy.decide("q", "q1", docs[:1], [], None).context_doc_ids == ("c0",)
