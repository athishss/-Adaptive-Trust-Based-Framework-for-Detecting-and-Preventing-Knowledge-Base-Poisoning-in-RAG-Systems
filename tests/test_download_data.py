from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import download_data  # noqa: E402


def test_full_corpus_query_sample_is_deterministic_and_bounded():
    gold = {f"q{i:04d}": {f"doc{i}"} for i in range(1000)}
    first = download_data.sample_query_ids(gold, 125, seed=17)
    second = download_data.sample_query_ids(gold, 125, seed=17)
    assert first == second
    assert len(first) == 125
    assert len(set(first)) == 125
    assert set(first) <= set(gold)
    assert len(download_data.sample_query_ids(gold, 2000, seed=17)) == len(gold)
    with pytest.raises(ValueError, match="positive"):
        download_data.sample_query_ids(gold, 0, seed=17)
