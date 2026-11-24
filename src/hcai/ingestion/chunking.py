"""Chunking strategies.

- fixed   : sliding word windows over the whole document (the common baseline).
- section : split on markdown headings first, then window inside each section, and prefix
            every chunk with "<title> > <section>". Keeps related facts together and puts the
            patient name / guideline topic into every chunk, which helps both BM25 and embeddings.
"""

from __future__ import annotations

import re

from hcai.ingestion.documents import Chunk, Document


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_fixed(doc: Document, size: int, overlap: int) -> list[Chunk]:
    words = doc.text.split()
    return [
        Chunk(f"{doc.doc_id}#{i}", doc.doc_id, doc.title, "", " ".join(w), doc.source_type, dict(doc.metadata))
        for i, w in enumerate(_windows(words, size, overlap))
    ]


_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.M)


def chunk_by_section(doc: Document, size: int, overlap: int) -> list[Chunk]:
    parts = _HEADING.split(doc.text)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("Overview", parts[0]))
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else ""))

    chunks: list[Chunk] = []
    header_line = doc.title.split(" (Illustrative")[0]
    for heading, body in sections:
        # Drop blockquote disclaimers; they add noise to every retrieval.
        body = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith(">")).strip()
        if not body:
            continue
        for w in _windows(body.split(), size, overlap):
            text = f"{header_line} > {heading}\n{' '.join(w)}"
            chunks.append(Chunk(f"{doc.doc_id}#{len(chunks)}", doc.doc_id, doc.title, heading, text, doc.source_type, dict(doc.metadata)))
    return chunks or chunk_fixed(doc, size, overlap)


def chunk_documents(docs: list[Document], strategy: str = "section", size: int = 120, overlap: int = 20) -> list[Chunk]:
    fn = chunk_by_section if strategy == "section" else chunk_fixed
    out: list[Chunk] = []
    for d in docs:
        out.extend(fn(d, size, overlap))
    return out
