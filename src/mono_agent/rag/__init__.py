"""Document loading, chunking, embeddings, hybrid retrieval, reranking."""

import logging

from qdrant_client import QdrantClient

from mono_agent.config import Settings
from mono_agent.rag.chunking import Strategy, chunk
from mono_agent.rag.embeddings import get_embeddings
from mono_agent.rag.loaders import load_dir
from mono_agent.rag.retriever import (
    CrossEncoderReranker,
    Retriever,
    collection_name,
    index_chunks,
)

log = logging.getLogger(__name__)


def index_docs(settings: Settings, strategies: list[Strategy]) -> dict[str, int]:
    """Load data/docs, chunk with each strategy, embed and store. Returns chunks per collection."""
    docs = load_dir(settings.docs_dir)
    client = QdrantClient(settings.qdrant_url)
    embeddings = get_embeddings(settings.embeddings)
    counts = {}
    for strategy in strategies:
        chunks = chunk(docs, strategy)
        name = collection_name(strategy, settings.embeddings)
        index_chunks(client, name, chunks, embeddings)
        counts[name] = len(chunks)
    return counts


def build_retriever(settings: Settings) -> Retriever | None:
    """None if Qdrant is down or the index is missing: the assistant still answers
    transaction questions and says it has no documents for the rest."""
    name = collection_name(settings.chunking, settings.embeddings)
    try:
        client = QdrantClient(settings.qdrant_url, timeout=5)
        if not client.collection_exists(name):
            log.warning("collection %s missing: run `mono-agent index`", name)
            return None
    except Exception as exc:  # connection refused, timeout, ...
        log.warning("Qdrant unavailable at %s: %s", settings.qdrant_url, exc)
        return None
    reranker = CrossEncoderReranker(settings.reranker) if settings.rerank else None
    return Retriever(client, name, get_embeddings(settings.embeddings), reranker)


def retrieve_fn(retriever: Retriever, settings: Settings):
    def retrieve(query: str) -> list[dict]:
        hits = retriever.search(
            query, k=settings.top_k, mode=settings.retrieval_mode, rerank=settings.rerank
        )
        return [{**h.chunk.payload(), "score": round(h.score, 4)} for h in hits]

    return retrieve
