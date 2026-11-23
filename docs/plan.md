# mono-agent: plan

A LangGraph multi-agent assistant for personal finances on top of the monobank Personal API.
Runs fully on synthetic data by default; real API is opt-in via `MONO_TOKEN`.

## Repo layout

```
src/mono_agent/
  config.py          # pydantic-settings: all env config in one place
  mono/              # monobank API client: rate limiter, window split, paging, retries, SQLite cache
  synthetic/         # deterministic statement generator (seeded)
  tools/             # Pydantic-validated tools for the transactions agent (parametrized SQL, no free-form SQL)
  graph/             # LangGraph state, nodes (router, transactions, knowledge, composer, guardrail), checkpointer
  rag/               # loaders (PDF/HTML), chunkers, embeddings, Qdrant, BM25, RRF, reranker
  providers/         # chat model factory per node, cost/latency tracking, fallbacks
  service/           # FastAPI (streaming), CLI, Streamlit UI
eval/
  dataset/           # 60+ questions with ground truth
  run.py             # one command -> eval/report.md
data/docs/           # public mono tariff/terms docs (you add real ones; placeholders shipped)
tests/
docs/                # plan.md, decisions.md, interview-notes.md
```

## Status

- M1–M3 done. M4 done: 77-question eval, provider comparison, RAG ablation, CI replay ([eval/report.md](../eval/report.md)).
- M5 done: FastAPI SSE service, Ukrainian Streamlit UI, provider fallbacks, JSON logs, self-hosted Langfuse v4, Docker Compose, GitHub Actions.
- M6 done: README (demo GIF, diagrams, results, 3-command quickstart), interview notes, decisions index.
- M7 done: MCP server for the tools (D52); QLoRA router vs prompting ([finetune/report.md](../finetune/report.md), D53).

## Milestones

| # | Scope | Done when |
|---|-------|-----------|
| 1 | Skeleton, synthetic generator, API client (token bucket, 31-day split, 500-item paging, backoff on 429/5xx, SQLite cache), PAN/IBAN masking | `uv run pytest` green; generator deterministic by seed; client tested against a fake HTTP transport |
| 2 | Transactions tools + ReAct agent + router + composer + guardrail; LangGraph state, SQLite checkpointer, HITL interrupt for "export report" | Graph tests with a fake LLM pass; CLI answers questions on synthetic data |
| 3 | RAG: loaders, fixed-size vs structure-aware chunking, local + API embeddings, Qdrant, BM25 + dense + RRF, cross-encoder rerank, each stage switchable | Knowledge agent answers with chunk citations; retrieval tests pass |
| 4 | Eval: dataset (60+), metrics (routing, tool name+args, numeric exact, recall@k/MRR, LLM-judge faithfulness, cost, latency), report.md with provider table + RAG ablation | `uv run python -m eval.run --config <provider>` writes `eval/report.md`; results committed |
| 5 | FastAPI streaming, Streamlit UI, Langfuse in Compose, structured logging, timeouts/fallbacks, GitHub Actions (ruff, pyright, pytest, eval smoke on recorded responses) | `docker compose up` brings the whole stack; CI green with no network |
| 6 | README (GIF, Mermaid, results), docs/decisions.md complete, docs/interview-notes.md | You can walk through every decision |
| 7 | Stretch: QLoRA router/categorizer vs prompting; MCP server for tools | Compared in eval report |

Each milestone ends with: what was done, what is not tested, next step.

## Initial technical choices (details and alternatives in decisions.md)

- **uv + Python 3.12**, `src/` layout, ruff + pyright + pytest.
- **SQLite** for transaction cache, checkpointer and the query layer: one file, zero ops, fits a single user.
- **Embeddings**: `BAAI/bge-m3` local (strong on Ukrainian, 8k context, multilingual) vs one API model (Gemini or OpenAI-compatible via OpenRouter); measured in the ablation, not assumed.
- **Reranker**: `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder, same family as embeddings).
- **Local LLM**: Ollama with Qwen (already installed here). API: Anthropic, Gemini, OpenRouter.
- **UI**: Streamlit (simple chat + streaming, stable). Chainlit was the alternative; skipped because its maintenance moved to the community in 2025.
- **Langfuse v3 self-hosted** needs Postgres, ClickHouse, Redis and MinIO; heavier than the rest of the stack. Tracing will be optional via env so the app runs without it.

## Open questions for you

1. Real API mode: do you want it exercised at all before milestone 5, or keep it to unit tests with a fake transport?
2. Which API providers do you have keys for (Anthropic / Gemini / OpenRouter)? The eval table depends on it.
3. Local model size your machine handles comfortably (e.g. `qwen3:8b` vs `qwen3:4b`)?
