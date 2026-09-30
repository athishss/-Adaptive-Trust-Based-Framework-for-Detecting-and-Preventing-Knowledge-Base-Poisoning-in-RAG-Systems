"""Document parsers: PDF, TXT, Markdown, HTML, and BEIR JSONL.

Optional dependencies are imported lazily so the core package installs with
numpy + scikit-learn only.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, Iterator

from ..utils.textnorm import normalise

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".jsonl"}
BEIR_SUFFIXES = {".jsonl", ".json", ".parquet"}

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


def _beir_row(raw: Dict[str, object], where: str) -> Dict[str, str]:
    doc_id = str(raw.get("_id") or raw.get("id") or "").strip()
    if not doc_id:
        raise ParseError(f"{where}: row without _id")
    return {
        "doc_id": doc_id,
        "title": normalise(str(raw.get("title") or "")),
        "text": normalise(str(raw.get("text") or "")),
    }


def iter_beir_parquet(path: str | Path, batch_size: int = 10_000) -> Iterator[Dict[str, str]]:
    """Stream a BEIR corpus/queries Parquet file (what Hugging Face serves).

    Read in row-group batches so a 764 MB corpus file does not have to be
    materialised in memory.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ParseError(
            "reading Parquet needs pyarrow: pip install 'trace-rag[data]' "
            "(or download the JSONL version of the corpus)"
        ) from exc
    parquet_file = pq.ParquetFile(str(path))
    available = set(parquet_file.schema_arrow.names)
    id_column = "_id" if "_id" in available else ("id" if "id" in available else None)
    if id_column is None:
        raise ParseError(f"{path}: Parquet file has no _id column (found {sorted(available)})")
    columns = [c for c in (id_column, "title", "text") if c in available]
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        for i, raw in enumerate(rows):
            if id_column != "_id":
                raw["_id"] = raw.get(id_column)
            yield _beir_row(raw, f"{path}:row{i}")


def iter_beir_jsonl(path: str | Path) -> Iterator[Dict[str, str]]:
    """Stream a BEIR ``corpus.jsonl`` ({_id, title, text} per line)."""
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"{path}:{lineno}: invalid JSON") from exc
            yield _beir_row(raw, f"{path}:{lineno}")


def iter_beir_corpus(path: str | Path) -> Iterator[Dict[str, str]]:
    """Stream a BEIR corpus from JSONL or Parquet.

    Hugging Face serves ``corpus-00000-of-00001.parquet``; the zip mirror ships
    ``corpus.jsonl``.  Both are accepted so either download works.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        yield from iter_beir_parquet(path)
        return
    if suffix not in BEIR_SUFFIXES:
        raise ParseError(f"unsupported BEIR corpus format: {suffix} (expected .jsonl or .parquet)")
    yield from iter_beir_jsonl(path)


def iter_files(root: str | Path, recursive: bool = True) -> Iterator[Path]:
    root_path = Path(root)
    pattern = "**/*" if recursive else "*"
    for path in sorted(root_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
