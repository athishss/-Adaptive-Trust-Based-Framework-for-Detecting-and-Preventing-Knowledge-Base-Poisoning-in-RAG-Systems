from __future__ import annotations

import pytest

from trace_rag.config import GenerationConfig
from trace_rag.contracts import Citation, LLMResponse
from trace_rag.generation import (GroundedGenerator, StubLLM, citation_precision, check_citations,
                                  evidence_mass, parse_citations)
from trace_rag.generation.prompts import build_answer_prompt, build_single_doc_prompt
from conftest import make_doc


class ScriptedLLM:
    name = "scripted"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 256, stop=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text, llm_calls=1, latency_ms=1.0, model=self.name)


def test_prompt_contains_ids_and_question():
    docs = [make_doc("d1", "Alpha text"), make_doc("d2", "Beta text")]
    prompt = build_answer_prompt("What is alpha?", docs)
    assert "[d1]" in prompt and "[d2]" in prompt and "What is alpha?" in prompt
    assert "INSUFFICIENT EVIDENCE" in prompt
    assert "[d1]" in build_single_doc_prompt("What is alpha?", docs[0])


def test_parse_citations_maps_only_known_ids():
    docs = [make_doc("d1"), make_doc("d2")]
    citations, invalid, cleaned = parse_citations("Alpha is a letter [d1]. Beta too [d9].", docs)
    assert [c.doc_id for c in citations] == ["d1"]
    assert invalid == ["d9"]
    assert "[" not in cleaned


def test_parse_citations_handles_multiple_ids_in_one_bracket():
    docs = [make_doc("d1"), make_doc("d2")]
    citations, invalid, _ = parse_citations("Fact [d1, d2].", docs)
    assert {c.doc_id for c in citations} == {"d1", "d2"} and invalid == []


def test_evidence_mass_compounds_across_sources_only():
    same_source = [make_doc("a", trust=0.5, source_id="s1"), make_doc("b", trust=0.5, source_id="s1")]
    two_sources = [make_doc("a", trust=0.5, source_id="s1"), make_doc("b", trust=0.5, source_id="s2")]
    citations = [Citation("a", "a", "x"), Citation("b", "b", "y")]
    assert evidence_mass(citations, same_source) == pytest.approx(0.5)
    assert evidence_mass(citations, two_sources) == pytest.approx(0.75)


def test_evidence_mass_is_zero_without_citations():
    assert evidence_mass([], [make_doc("a")]) == 0.0


def test_generator_answers_with_citation():
    docs = [make_doc("d1", "Gustave Eiffel designed the Eiffel Tower in 1889.", trust=0.9)]
    outcome = GroundedGenerator(StubLLM(), GenerationConfig()).generate(
        "Who designed the Eiffel Tower?", "q1", docs)
    assert not outcome.record.abstained
    assert outcome.record.used_doc_ids == ("d1",)
    assert outcome.record.evidence_mass == pytest.approx(0.9)


def test_generator_abstains_without_context():
    outcome = GroundedGenerator(StubLLM(), GenerationConfig()).generate("q?", "q1", [])
    assert outcome.record.abstained and outcome.record.abstain_reason == "no_context"
    assert outcome.record.llm_calls == 0            # abstention must not cost an LLM call


def test_generator_abstains_on_low_trust_evidence():
    docs = [make_doc("d1", "Zog the Alien designed the Eiffel Tower.", trust=0.2)]
    config = GenerationConfig(abstain_evidence_mass=0.5)
    outcome = GroundedGenerator(StubLLM(), config).generate("Who designed the Eiffel Tower?", "q1", docs)
    assert outcome.record.abstained
    assert outcome.record.abstain_reason == "insufficient_evidence"


def test_generator_abstains_when_model_says_so():
    docs = [make_doc("d1", "Unrelated text about plants.")]
    outcome = GroundedGenerator(ScriptedLLM("INSUFFICIENT EVIDENCE"), GenerationConfig()).generate(
        "Who designed the Eiffel Tower?", "q1", docs)
    assert outcome.record.abstained and outcome.record.abstain_reason == "model_abstained"


def test_generator_abstains_on_hallucinated_citations():
    docs = [make_doc("d1", "Some text", trust=0.9)]
    outcome = GroundedGenerator(ScriptedLLM("The answer is 42 [made_up_id]."),
                                GenerationConfig()).generate("q?", "q1", docs)
    assert outcome.record.abstained and outcome.record.abstain_reason == "invalid_citations"
    assert outcome.invalid_citations == ("made_up_id",)


def test_generator_abstains_when_citations_missing():
    docs = [make_doc("d1", "Some text", trust=0.9)]
    outcome = GroundedGenerator(ScriptedLLM("The answer is 42."), GenerationConfig()).generate(
        "q?", "q1", docs)
    assert outcome.record.abstained and outcome.record.abstain_reason == "no_citations"


def test_citation_checker_marks_support():
    docs = [make_doc("d1", "Gustave Eiffel designed the tower.", trust=0.9)]

    def checker(premise: str, hypothesis: str):
        return ("Eiffel" in premise and "Eiffel" in hypothesis), 0.91

    outcome = GroundedGenerator(StubLLM(), GenerationConfig(), citation_checker=checker).generate(
        "Who designed the tower?", "q1", docs)
    assert all(c.supported is True for c in outcome.record.citations)
    assert citation_precision(outcome.record.citations) == 1.0


def test_generator_abstains_when_every_citation_is_unsupported():
    docs = [make_doc("d1", "Gustave Eiffel designed the tower.", trust=0.9)]
    outcome = GroundedGenerator(StubLLM(), GenerationConfig(),
                                citation_checker=lambda p, h: (False, 0.02)).generate(
        "Who designed the tower?", "q1", docs)
    assert outcome.record.abstained and outcome.record.abstain_reason == "citations_unsupported"


def test_check_citations_without_checker_leaves_state_unknown():
    citations = [Citation("d1", "d1", "sentence")]
    assert check_citations(citations, [make_doc("d1")])[0].supported is None
    assert citation_precision(citations) is None


def test_context_is_capped_by_config():
    docs = [make_doc(f"d{i}", f"text {i}", trust=0.9) for i in range(10)]
    llm = ScriptedLLM("Answer [d0].")
    GroundedGenerator(llm, GenerationConfig(max_context_docs=3)).generate("q?", "q1", docs)
    assert llm.calls == 1
