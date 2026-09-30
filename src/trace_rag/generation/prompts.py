"""Prompts.  Kept in one file so every prompt used in the paper is auditable."""

from __future__ import annotations

from typing import Sequence

from ..contracts import RetrievedDocument

SYSTEM_INSTRUCTION = (
    "You answer strictly from the numbered context passages. "
    "Every sentence you write must end with the identifier of the passage that supports it, "
    "in square brackets, for example [doc_12#0003]. "
    "Never cite a passage that is not in the context. "
    "If the context does not contain the answer, reply with exactly: "
    "INSUFFICIENT EVIDENCE"
)

ANSWER_TEMPLATE = """{system}

Context passages:
{context}

Question: {question}

Answer (short, with a [passage-id] citation after every sentence):"""

SINGLE_DOC_TEMPLATE = """{system}

Context passages:
[{doc_id}] {text}

Question: {question}

Answer using only the passage above:"""


def format_context(documents: Sequence[RetrievedDocument], max_chars_per_doc: int = 1200) -> str:
    """One block per passage, prefixed with the id the model must cite."""
    blocks = []
    for doc in documents:
        text = doc.text if len(doc.text) <= max_chars_per_doc else doc.text[:max_chars_per_doc] + " ..."
        blocks.append(f"[{doc.doc_id}] {text}")
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, documents: Sequence[RetrievedDocument],
                        system: str = SYSTEM_INSTRUCTION, max_chars_per_doc: int = 1200) -> str:
    return ANSWER_TEMPLATE.format(system=system, context=format_context(documents, max_chars_per_doc),
                                  question=question.strip())


def build_single_doc_prompt(question: str, document: RetrievedDocument,
                            system: str = SYSTEM_INSTRUCTION) -> str:
    """Used by Person B's verifier for the "what does this passage alone claim" step."""
    return SINGLE_DOC_TEMPLATE.format(system=system, doc_id=document.doc_id,
                                      text=document.text, question=question.strip())
