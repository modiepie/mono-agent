"""Retrieval pipeline with switchable stages, so the eval can measure each one:

dense (Qdrant)  ─┐
                 ├─ RRF fusion (hybrid) ─> cross-encoder rerank (optional) ─> top k
BM25 (in memory)─┘
"""

import re
import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from mono_agent.rag.bm25 import BM25
from mono_agent.rag.chunking import Chunk, Strategy

Mode = Literal["dense", "bm25", "hybrid"]


def collection_name(strategy: Strategy, embeddings_spec: str) -> str:
    return f"mono_docs_{strategy}_{re.sub(r'[^a-z0-9]+', '_', embeddings_spec.lower())}"


def index_chunks(
    client: QdrantClient, collection: str, chunks: list[Chunk], embeddings: Embeddings
) -> None:
    """(Re)create the collection: indexing is cheap at this size and avoids stale chunks."""
    vectors = embeddings.embed_documents([c.embed_text for c in chunks])
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection, VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
    )
    client.upsert(
        collection,
        [
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)), vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = Σ 1 / (k + rank of d in each list), rank from 1.
    Uses ranks only, so BM25 and cosine scores never need to be on the same scale."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class Reranker(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder  # heavy import: only when used

        from mono_agent.rag.embeddings import load_cached

        self.model = load_cached(CrossEncoder, model)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return self.model.predict([(query, t) for t in texts]).tolist()


@dataclass
class Hit:
    chunk: Chunk
    score: float
    stage: str  # which stage produced the final order (for debugging / traces)


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embeddings: Embeddings,
        reranker: Reranker | None = None,
    ) -> None:
        self.client, self.collection = client, collection
        self.embeddings, self.reranker = embeddings, reranker
        # BM25 is built from the same chunks Qdrant holds: one source of truth.
        points, _ = client.scroll(collection, limit=10_000, with_payload=True)
        self.chunks = {p.payload["id"]: Chunk(**p.payload) for p in points if p.payload}
        self.order = list(self.chunks)
        self.bm25 = BM25([self.chunks[i].embed_text for i in self.order])

    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = self.embeddings.embed_query(query)
        res = self.client.query_points(self.collection, query=vector, limit=k, with_payload=True)
        return [(p.payload["id"], p.score) for p in res.points if p.payload]

    def sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(self.order[i], s) for i, s in self.bm25.top(query, k)]

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid",
        rerank: bool = True,
        candidates: int = 20,
    ) -> list[Hit]:
        if mode == "dense":
            ranked, stage = self.dense(query, candidates), "dense"
        elif mode == "bm25":
            ranked, stage = self.sparse(query, candidates), "bm25"
        else:
            fused = rrf(
                [
                    [i for i, _ in self.dense(query, candidates)],
                    [i for i, _ in self.sparse(query, candidates)],
                ]
            )
            ranked, stage = fused[:candidates], "hybrid"
        if rerank and self.reranker and ranked:
            texts = [self.chunks[i].embed_text for i, _ in ranked]
            ranked = sorted(
                zip([i for i, _ in ranked], self.reranker.scores(query, texts), strict=True),
                key=lambda x: -x[1],
            )
            stage += "+rerank"
        return [Hit(self.chunks[i], float(s), stage) for i, s in ranked[:k]]
