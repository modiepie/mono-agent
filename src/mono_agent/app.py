"""Wire settings -> client, cache, tools, models, checkpointer -> compiled graph."""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from mono_agent.config import Settings
from mono_agent.graph.builder import Models, build_graph
from mono_agent.mono.cache import Cache
from mono_agent.mono.factory import build_client
from mono_agent.providers import chat_model
from mono_agent.rag import build_retriever, retrieve_fn
from mono_agent.tools.finance import ToolContext


def build_app(settings: Settings):
    ctx = ToolContext(build_client(settings), Cache(settings.cache_db), settings.today)
    models = Models(
        router=chat_model(settings.router_model),
        agent=chat_model(settings.agent_model),
        composer=chat_model(settings.composer_model),
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    retriever = build_retriever(settings)
    retrieve = retrieve_fn(retriever, settings) if retriever else None
    return build_graph(models, ctx, settings.data_dir / "exports", SqliteSaver(conn), retrieve)
