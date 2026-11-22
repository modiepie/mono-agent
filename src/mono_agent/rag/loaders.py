"""Load HTML and PDF documents into an ordered list of blocks with their heading path."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from mono_agent.guardrails import strip_injection_sentences

log = logging.getLogger(__name__)


@dataclass
class Block:
    headings: tuple[str, ...]  # e.g. ("Кредитний ліміт", "Пільговий період до 62 днів")
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    date: str | None
    sample: bool
    blocks: list[Block] = field(default_factory=list)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _add(doc: Document, headings: tuple[str, ...], text: str) -> None:
    text, removed = strip_injection_sentences(_clean(text))
    if removed:
        log.warning("removed instruction-like text from %s", doc.doc_id)
    if text:
        doc.blocks.append(Block(headings, text))


def load_html(path: Path) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    def meta(name: str) -> str | None:
        tag = soup.find("meta", attrs={"name": name})
        return str(tag["content"]) if tag and tag.get("content") else None

    title = _clean(soup.title.get_text()) if soup.title else path.stem
    doc = Document(path.stem, title, path.name, meta("date"), meta("sample") == "true")
    stack: list[str] = []
    for el in (soup.body or soup).find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if el.name[0] == "h":
            level = int(el.name[1])
            stack = [*stack[: level - 1], _clean(el.get_text())]
        elif el.name == "tr":
            _add(doc, tuple(stack), " | ".join(c.get_text() for c in el.find_all(["td", "th"])))
        else:
            _add(doc, tuple(stack), el.get_text())
    return doc


# A numbered line ("2.1 Пільговий період") or a short ALL-CAPS line counts as a heading.
_PDF_HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+\S.{0,80}|[A-ZА-ЯІЇЄҐ0-9 ,'’-]{4,80})$")


def load_pdf(path: Path) -> Document:
    reader = PdfReader(path)
    info = reader.metadata
    title = (info.title if info and info.title else None) or path.stem
    date = info.creation_date.date().isoformat() if info and info.creation_date else None
    doc = Document(path.stem, title, path.name, date, sample=False)
    heading: tuple[str, ...] = (title,)
    paragraph: list[str] = []
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            if not line:
                _add(doc, heading, " ".join(paragraph))
                paragraph = []
            elif _PDF_HEADING.match(line):
                _add(doc, heading, " ".join(paragraph))
                paragraph = []
                heading = (title, line)
            else:
                paragraph.append(line)
    _add(doc, heading, " ".join(paragraph))
    return doc


def load_dir(path: Path) -> list[Document]:
    loaders = {".html": load_html, ".htm": load_html, ".pdf": load_pdf}
    return [
        loaders[p.suffix.lower()](p) for p in sorted(path.iterdir()) if p.suffix.lower() in loaders
    ]
