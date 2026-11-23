"""Wire settings -> client, cache, tools, models, checkpointer -> compiled graph."""

import sqlite3
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Checkpointer

from mono_agent.config import Settings
from mono_agent.graph.builder import Models, Retrieve, build_graph
from mono_agent.mono.cache import Cache
from mono_agent.mono.factory import build_client
from mono_agent.providers import chat_model
from mono_agent.rag import build_retriever, retrieve_fn
from mono_agent.tools.finance import ToolContext


def models_from(settings: Settings) -> Models:
    return Models(
        router=chat_model(settings.router_model),
        agent=chat_model(settings.agent_model),
        composer=chat_model(settings.composer_model),
    )


def build_app(
    settings: Settings,
    models: Models | None = None,
    checkpointer: Checkpointer | None = None,
    retrieve: Retrieve | Literal["auto"] | None = "auto",
    export_dir: Path | None = None,
    cache_path: Path | str | None = None,
):
    """Everything comes from settings unless passed in (the eval injects its own parts)."""
    ctx = ToolContext(
        build_client(settings), Cache(cache_path or settings.cache_db), settings.today
    )
    if checkpointer is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    if retrieve == "auto":
        retriever = build_retriever(settings)
        retrieve = retrieve_fn(retriever, settings) if retriever else None
    return build_graph(
        models or models_from(settings),
        ctx,
        export_dir or settings.data_dir / "exports",
        checkpointer,
        retrieve,
    )
