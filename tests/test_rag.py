"""RAG pipeline offline: fake embeddings (hashed bag of words) + Qdrant in-memory mode."""

import hashlib
import math
from itertools import pairwise
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from mono_agent.rag.bm25 import BM25, tokenize
from mono_agent.rag.chunking import chunk, fixed_chunks, structure_chunks
from mono_agent.rag.loaders import Block, Document, load_dir, load_html, load_pdf
from mono_agent.rag.retriever import Retriever, collection_name, index_chunks, rrf

DOCS = Path(__file__).parent.parent / "data" / "docs"


class HashEmbeddings(Embeddings):
    """Deterministic stand-in for a real model: similar words -> similar vectors."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 128
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 128] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    """Scores by how many query tokens appear in the text, like a (very dumb) cross-encoder."""

    def scores(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) for t in texts]


@pytest.fixture(scope="module")
def docs():
    return load_dir(DOCS)


# --- loading -----------------------------------------------------------------------------


def test_html_loader_keeps_heading_path_and_metadata():
    doc = load_html(DOCS / "credit-limit.html")
    assert doc.title == "Кредитний ліміт і пільговий період"
    assert doc.date == "2025-08-15" and doc.sample
    example = next(b for b in doc.blocks if "3 березня" in b.text)
    assert example.headings[-2:] == ("Пільговий період до 62 днів", "Приклад розрахунку")


def test_html_tables_become_rows():
    doc = load_html(DOCS / "tariffs.html")
    assert any(b.text.startswith("Перекази за рахунок кредитних коштів | 4%") for b in doc.blocks)


def test_injection_sentence_is_dropped_but_paragraph_kept():
    doc = load_html(DOCS / "faq-en.html")
    text = " ".join(b.text for b in doc.blocks)
    assert "ignore all previous instructions" not in text.lower()
    assert "A late fee of 100 UAH is charged" in text


def _minimal_pdf(lines: list[str]) -> bytes:
    """A valid one-page PDF with Helvetica text, written by hand to avoid a PDF library."""
    ops = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({t}) Tj T*" for t in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(ops)} >>\nstream\n{ops}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return out.encode("latin-1")


def test_pdf_loader_detects_numbered_headings(tmp_path):
    pdf = tmp_path / "terms.pdf"
    pdf.write_bytes(
        _minimal_pdf(
            [
                "1. Grace period",
                "Up to 62 days without interest.",
                "2. Fees",
                "Cash withdrawal costs 4%.",
            ]
        )
    )
    doc = load_pdf(pdf)
    by_heading = {b.headings[-1]: b.text for b in doc.blocks}
    assert by_heading["1. Grace period"] == "Up to 62 days without interest."
    assert by_heading["2. Fees"] == "Cash withdrawal costs 4%."


# --- chunking ----------------------------------------------------------------------------


def test_fixed_chunks_respect_size_and_overlap():
    words = " ".join(f"слово{i}" for i in range(400))
    doc = Document("d", "T", "d.html", None, False, [Block(("T",), words)])
    chunks = fixed_chunks(doc, size=300, overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert len(chunks) > 5
    for a, b in pairwise(chunks):
        assert a.text.split()[-1] in b.text  # consecutive chunks overlap
    joined = " ".join(c.text for c in chunks)
    assert all(f"слово{i}" in joined for i in range(400))  # nothing lost


def test_structure_chunks_follow_sections_and_carry_context(docs):
    chunks = [c for c in structure_chunks(next(d for d in docs if d.doc_id == "credit-limit"))]
    example = next(c for c in chunks if "3 березня" in c.text)
    assert example.section == "Пільговий період до 62 днів > Приклад розрахунку"
    assert example.embed_text.startswith("Кредитний ліміт і пільговий період > Пільговий")
    assert "мінімальний платіж" not in example.text.lower()  # no bleed from other sections


def test_structure_chunks_split_long_sections():
    long = " ".join(["Речення про кешбек."] * 200)
    doc = Document("d", "T", "d.html", None, False, [Block(("T", "S"), long)])
    chunks = structure_chunks(doc, max_chars=500)
    assert len(chunks) > 1 and all(len(c.text) <= 500 for c in chunks)
    assert {c.section for c in chunks} == {"S"}


def test_chunk_ids_are_unique_per_strategy(docs):
    for strategy in ("fixed", "structure"):
        ids = [c.id for c in chunk(docs, strategy)]
        assert len(ids) == len(set(ids))


# --- retrieval ---------------------------------------------------------------------------


def test_bm25_ranks_matching_chunk_first_and_handles_inflection():
    bm = BM25(["кешбек нараховується щомісяця", "кредитний ліміт", "банка для накопичень"])
    assert bm.top("кешбеку", 3)[0][0] == 0  # inflected form matches via 6-char stems
    assert bm.top("weather", 3) == []


def test_rrf_rewards_agreement_between_rankings():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == fused["b"] > fused["c"] == fused["d"]
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)


@pytest.fixture(scope="module")
def retriever(docs):
    client = QdrantClient(":memory:")
    name = collection_name("structure", "local:fake")
    index_chunks(client, name, chunk(docs, "structure"), HashEmbeddings())
    return Retriever(client, name, HashEmbeddings(), OverlapReranker())


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("rerank", [False, True])
def test_every_mode_finds_the_grace_period_section(retriever, mode, rerank):
    hits = retriever.search("пільговий період кредитний ліміт", k=3, mode=mode, rerank=rerank)
    assert any(h.chunk.section.startswith("Пільговий період") for h in hits)
    assert hits[0].stage == mode + ("+rerank" if rerank else "")


def test_reranker_decides_final_order(retriever):
    class PreferJars:
        def scores(self, query, texts):
            return [1.0 if "Банка" in t else 0.0 for t in texts]

    query = "кешбек і банки"
    candidates = [h.chunk.id for h in retriever.search(query, k=20, mode="hybrid", rerank=False)]
    jar = next(i for i in candidates if i.startswith("jars"))
    assert candidates[0] != jar  # not first before reranking...
    with_jars = Retriever(
        retriever.client, retriever.collection, retriever.embeddings, PreferJars()
    )
    top = with_jars.search(query, k=3, mode="hybrid", rerank=True)
    assert top[0].chunk.doc_id == "jars" and top[0].stage == "hybrid+rerank"  # ...first after


def test_collection_name_is_safe_and_distinct():
    a = collection_name("fixed", "local:BAAI/bge-m3")
    b = collection_name("structure", "google_genai:gemini-embedding-001")
    assert a == "mono_docs_fixed_local_baai_bge_m3" and a != b
