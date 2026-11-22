"""Two chunking strategies, compared in the eval's RAG ablation.

- fixed: sliding window of ~600 characters with 100 overlap over the whole document text,
  blind to structure. The common baseline.
- structure: one chunk per section (split further at paragraph boundaries when long), and the
  embedded text starts with "Title > Section" so a chunk still says what it is about.
"""

import re
from dataclasses import asdict, dataclass
from typing import Literal

from mono_agent.rag.loaders import Document

Strategy = Literal["fixed", "structure"]


@dataclass
class Chunk:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    section: str
    date: str | None
    source: str
    sample: bool
    text: str  # shown to the LLM and cited
    embed_text: str  # what gets embedded / BM25-indexed

    def payload(self) -> dict:
        return asdict(self)


def _section(doc: Document, headings: tuple[str, ...]) -> str:
    # The h1 usually repeats the title; keep the path below it.
    path = [h for h in headings if h != doc.title]
    return " > ".join(path) or doc.title


def fixed_chunks(doc: Document, size: int = 600, overlap: int = 100) -> list[Chunk]:
    text, starts = "", []  # starts[i] = (offset where block i begins, block)
    for block in doc.blocks:
        starts.append((len(text), block))
        text += block.text + "\n"
    chunks, pos = [], 0
    while pos < len(text.rstrip()):
        end = min(pos + size, len(text))
        if end < len(text):  # don't cut a word in half
            space = text.rfind(" ", pos + size // 2, end)
            end = space if space > 0 else end
        piece = text[pos:end].strip()
        block = next(b for off, b in reversed(starts) if off <= pos)  # section at chunk start
        chunks.append(
            Chunk(
                id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=_section(doc, block.headings),
                date=doc.date,
                source=doc.source,
                sample=doc.sample,
                text=piece,
                embed_text=piece,  # no added context: that is the point of the baseline
            )
        )
        if end >= len(text):
            break
        pos = max(end - overlap, pos + 1)
    return chunks


def structure_chunks(doc: Document, max_chars: int = 1000) -> list[Chunk]:
    sections: dict[tuple[str, ...], list[str]] = {}
    for block in doc.blocks:
        sections.setdefault(block.headings, []).append(block.text)

    chunks = []
    for headings, texts in sections.items():
        section = _section(doc, headings)
        parts, current = [], ""
        for t in _split_long(texts, max_chars):
            if current and len(current) + len(t) + 1 > max_chars:
                parts.append(current)
                current = ""
            current = f"{current}\n{t}".strip()
        parts.append(current)
        for part in parts:
            chunks.append(
                Chunk(
                    id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    date=doc.date,
                    source=doc.source,
                    sample=doc.sample,
                    text=part,
                    embed_text=f"{doc.title} > {section}\n{part}",
                )
            )
    return chunks


def _split_long(texts: list[str], max_chars: int) -> list[str]:
    """Paragraphs longer than max_chars are split at sentence boundaries."""
    out = []
    for t in texts:
        if len(t) <= max_chars:
            out.append(t)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", t):
            if current and len(current) + len(sentence) + 1 > max_chars:
                out.append(current)
                current = ""
            current = f"{current} {sentence}".strip()
        out.append(current)
    return out


def chunk(docs: list[Document], strategy: Strategy) -> list[Chunk]:
    fn = fixed_chunks if strategy == "fixed" else structure_chunks
    return [c for d in docs for c in fn(d)]
