"""HTTP API: chat over Server-Sent Events, with human-in-the-loop resume.

    POST /chat                      {"message": "...", "thread_id": "optional"}  -> SSE
    POST /chat/{thread_id}/resume   {"approve": true}                            -> SSE
    GET  /health

SSE events (event name + JSON data):
    thread   {"thread_id"}                         first event: id to continue the conversation
    step     {"node", "detail"}                    progress: routing, each tool call, retrieval...
    confirm  {"action", "question", ...}           graph paused; answer with /resume
    answer   {"text", "citations", "tools", ...}   final answer, already number-checked
    error    {"message"}

Only progress is streamed, not answer tokens: the composer's draft may be rejected by the
number check and rewritten, so the user only ever sees the verified answer.
"""

import json
import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from mono_agent.config import Settings, get_settings
from mono_agent.observability import flush_traces, setup_logging, tracing_callbacks
from mono_agent.providers.usage import UsageTracker

log = logging.getLogger("mono_agent.api")
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID.pattern)


class ResumeRequest(BaseModel):
    approve: bool


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _step_detail(node: str, update: dict) -> dict:
    """What to show for a finished node; never the raw state (it can be large)."""
    if node == "router" and update:
        return {"intent": update.get("intent"), "plan": update.get("plan")}
    if node == "transactions_tools":
        last = (update.get("tool_results") or [{}])[-1]
        return {"tool": last.get("tool"), "args": last.get("args"), "ok": last.get("ok")}
    if node == "knowledge":
        return {"documents": [d["id"] for d in update.get("retrieved_docs", [])]}
    return {}


def _answer(values: dict) -> dict:
    return {
        "text": values["messages"][-1].content,
        "citations": values.get("citations", []),
        "tools": [
            {"tool": r["tool"], "args": r["args"], "ok": r["ok"]}
            for r in values.get("tool_results", [])
        ],
        "documents": [
            {"id": d["id"], "title": d["title"], "section": d["section"], "sample": d.get("sample")}
            for d in values.get("retrieved_docs", [])
        ],
        "export_path": values.get("export_path"),
    }


def create_api(build_graph=None, settings: Settings | None = None) -> FastAPI:
    """`build_graph` is injectable so tests run the real API on a graph with fake models."""
    settings = settings or get_settings()
    state: dict[str, Any] = {}
    # shortcut: one graph run at a time. The SQLite checkpointer and cache connections are
    # shared, and a local model serves one request at a time anyway. Replace with a
    # connection per request (or Postgres) before serving several users.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        if build_graph is not None:
            state["graph"] = build_graph()
        else:
            from mono_agent.app import build_app
            from mono_agent.rag import ensure_index

            ensure_index(settings)
            state["graph"] = build_app(settings)
        log.info(
            "api started",
            extra={
                "fields": {"real_data": settings.use_real_api, "tracing": settings.tracing_enabled}
            },
        )
        yield
        flush_traces(settings)

    app = FastAPI(title="mono-agent", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "data": "real" if settings.use_real_api else "synthetic",
            "today": str(settings.today),
            "models": {
                "router": settings.router_model,
                "agent": settings.agent_model,
                "composer": settings.composer_model,
                "fallback": settings.fallback_model or None,
            },
            "tracing": settings.tracing_enabled,
        }

    def execute(graph_input: Any, thread_id: str, emit) -> None:
        """Run the graph to the end and emit SSE events. Runs in its own thread, so a client
        that disconnects mid-stream can't leave the run (or the lock) half-finished."""
        tracker = UsageTracker()
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [tracker, *tracing_callbacks(settings)],
            "metadata": {"langfuse_session_id": thread_id},
        }
        graph = state["graph"]
        start = time.perf_counter()
        status = "ok"
        with lock:
            try:
                for chunk in graph.stream(graph_input, config, stream_mode="updates"):
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            status = "paused"
                            emit(_sse("confirm", update[0].value))
                        else:
                            emit(
                                _sse(
                                    "step",
                                    {"node": node, "detail": _step_detail(node, update or {})},
                                )
                            )
                if status != "paused":
                    emit(_sse("answer", _answer(graph.get_state(config).values)))
            except Exception:
                status = "error"
                log.exception("chat failed", extra={"fields": {"thread_id": thread_id}})
                emit(_sse("error", {"message": "internal error"}))
        latency = round(time.perf_counter() - start, 2)
        log.info(
            "chat",
            extra={
                "fields": {
                    "thread_id": thread_id,
                    "status": status,
                    "latency_s": latency,
                    **tracker.totals(),
                }
            },
        )

    def run(graph_input: Any, thread_id: str) -> Iterator[str]:
        events: queue.Queue[str | None] = queue.Queue()

        def worker() -> None:
            try:
                execute(graph_input, thread_id, events.put)
            finally:
                events.put(None)  # end of stream

        threading.Thread(target=worker, daemon=True).start()
        yield _sse("thread", {"thread_id": thread_id})
        while (event := events.get()) is not None:
            yield event

    def stream(events: Iterator[str]) -> StreamingResponse:
        # A sync generator: Starlette runs it in a worker thread, the graph itself is sync.
        return StreamingResponse(
            events, media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        thread_id = req.thread_id or uuid.uuid4().hex[:12]
        return stream(run({"messages": [HumanMessage(req.message)]}, thread_id))

    @app.post("/chat/{thread_id}/resume")
    def resume(thread_id: str, req: ResumeRequest) -> StreamingResponse:
        if not THREAD_ID.match(thread_id):
            raise HTTPException(422, "invalid thread id")
        snapshot = state["graph"].get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.interrupts:
            raise HTTPException(409, "nothing to confirm in this thread")
        return stream(run(Command(resume=req.approve), thread_id))

    return app


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    uvicorn.run(create_api(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
