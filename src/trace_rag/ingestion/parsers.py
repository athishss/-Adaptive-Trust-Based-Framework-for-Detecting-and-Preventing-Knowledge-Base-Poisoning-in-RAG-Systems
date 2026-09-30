"""Document parsers: PDF, TXT, Markdown, HTML, and BEIR JSONL.

Optional dependencies are imported lazily so the core package installs with
numpy + scikit-learn only.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..utils.textnorm import normalise

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".jsonl"}

_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_MARKS = re.compile(r"[*_`>#]{1,3}")


class ParseError(RuntimeError):
    pass


def parse_txt(path: Path) -> str:
    return normalise(path.read_text(encoding="utf-8", errors="replace"))


def parse_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_MARKS.sub("", text)
    return normalise(text)


def parse_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:                                        # prefer a real parser when present
        from bs4 import BeautifulSoup          # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return normalise(soup.get_text(" "))
    except ImportError:
        cleaned = _SCRIPT_STYLE.sub(" ", raw)
        return normalise(html.unescape(_TAG.sub(" ", cleaned)))


def parse_pdf(path: Path) -> str:
    try:
        import fitz  # type: ignore  # PyMuPDF, fastest when available

        with fitz.open(path) as doc:
            return normalise(" ".join(page.get_text("text") for page in doc))
    except ImportError:
        pass
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParseError(
            "PDF support needs either 'pymupdf' or 'pypdf' installed"
        ) from exc
    reader = PdfReader(str(path))
    return normalise(" ".join((page.extract_text() or "") for page in reader.pages))


def parse_file(path: str | Path) -> str:
    """Dispatch on suffix.  Raises ParseError for unsupported types."""
    p = Path(path)
    if not p.is_file():
        raise ParseError(f"not a file: {p}")
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return parse_txt(p)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(p)
    if suffix in {".html", ".htm"}:
        return parse_html(p)
    if suffix == ".pdf":
        return parse_pdf(p)
    raise ParseError(f"unsupported file type: {suffix}")


def iter_beir_corpus(path: str | Path) -> Iterator[Dict[str, str]]:
    """Stream a BEIR ``corpus.jsonl`` ({_id, title, text} per line).

    Tolerates blank lines; raises ParseError on malformed JSON so a broken
    download fails loudly instead of silently shrinking the corpus.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"{path}:{lineno}: invalid JSON") from exc
            doc_id = str(row.get("_id") or row.get("id") or "").strip()
            if not doc_id:
                raise ParseError(f"{path}:{lineno}: row without _id")
            yield {
                "doc_id": doc_id,
                "title": normalise(str(row.get("title", ""))),
                "text": normalise(str(row.get("text", ""))),
            }


def iter_files(root: str | Path, recursive: bool = True) -> Iterator[Path]:
    root_path = Path(root)
    pattern = "**/*" if recursive else "*"
    for path in sorted(root_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
