"""RAG ablation: retrieval only (no LLM), every question with evidence, every configuration.

Measures what each stage adds: BM25 vs dense vs hybrid (RRF), with and without the
cross-encoder reranker, for both chunking strategies.
"""

import statistics
import time

from qdrant_client import QdrantClient

from eval.dataset import Question
from eval.metrics import retrieval_scores
from eval.runner import eval_settings
from mono_agent.rag.embeddings import get_embeddings
from mono_agent.rag.retriever import CrossEncoderReranker, Retriever, collection_name


def _log(*args: object) -> None:
    print(*args, flush=True)  # visible live even when stdout is piped


STAGES = [("bm25", False), ("dense", False), ("hybrid", False), ("dense", True), ("hybrid", True)]


def run_ablation(questions: list[Question], log=_log) -> list[dict]:
    settings = eval_settings()
    client = QdrantClient(settings.qdrant_url)
    embeddings = get_embeddings(settings.embeddings)
    reranker = CrossEncoderReranker(settings.reranker)
    with_evidence = [q for q in questions if q.evidence]
    rows = []
    for strategy in ("fixed", "structure"):
        retriever = Retriever(
            client, collection_name(strategy, settings.embeddings), embeddings, reranker
        )
        for mode, rerank in STAGES:
            per_q, latencies = [], []
            for q in with_evidence:
                start = time.perf_counter()
                hits = retriever.search(q.q, k=10, mode=mode, rerank=rerank)  # type: ignore[arg-type]
                latencies.append(time.perf_counter() - start)
                chunks = [h.chunk.payload() for h in hits]
                per_q.append({k: retrieval_scores(q.evidence, chunks, k=k) for k in (1, 3, 5)})
            row = {
                "chunking": strategy,
                "mode": mode + ("+rerank" if rerank else ""),
                "questions": len(per_q),
                **{
                    f"recall@{k}": statistics.mean(s[k]["recall"] for s in per_q) for k in (1, 3, 5)
                },
                "mrr": statistics.mean(s[5]["mrr"] for s in per_q),
                "latency_ms": round(1000 * statistics.mean(latencies), 1),
                "misses@5": [
                    q.id for q, s in zip(with_evidence, per_q, strict=True) if s[5]["recall"] < 1
                ],
            }
            rows.append(row)
            log(
                f"{strategy:9} {row['mode']:14} recall@5={row['recall@5']:.2f} "
                f"mrr={row['mrr']:.2f} {row['latency_ms']}ms"
            )
    return rows
