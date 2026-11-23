"""MCP server exposing the seven finance tools to any MCP client (Claude Desktop, IDEs, ...).

    uv run mono-agent-mcp        # stdio transport

Same tools, validation and safety as inside the graph: run_tool validates the arguments with
the Pydantic input models, card numbers are masked at ingestion, and merchant names and comments
are sanitized inside the tools. Synthetic data unless MONO_TOKEN is set; with real data, the
MCP client's model sees your (masked) transactions.
"""

import asyncio
import json

import mcp_types as types
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from mono_agent.config import get_settings
from mono_agent.mono.cache import Cache
from mono_agent.mono.factory import build_client
from mono_agent.tools import TOOLS, run_tool
from mono_agent.tools.finance import ToolContext

# All tools only read: the bank API is read-only, and sync_statement only fills the local cache.
_READ_ONLY = types.ToolAnnotations(read_only_hint=True, destructive_hint=False)


def create_server(ctx: ToolContext) -> Server:
    tools = [
        types.Tool(
            name=spec.name,
            description=spec.description,
            input_schema=spec.schema()["function"]["parameters"],
            annotations=_READ_ONLY,
        )
        for spec in TOOLS.values()
    ]

    async def list_tools(_ctx, _params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        # shortcut: runs on the event loop (one client over stdio); a first real-account sync
        # blocks it for minutes. Move to a thread if this ever serves several clients.
        output, ok = run_tool(ctx, params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(output, ensure_ascii=False))],
            structured_content=output,
            is_error=not ok,
        )

    return Server("mono-agent", on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> None:
    load_dotenv()
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(build_client(settings), Cache(settings.cache_db), settings.today)
    server = create_server(ctx)

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(serve())


if __name__ == "__main__":
    main()
