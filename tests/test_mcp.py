import asyncio

from mcp import Client

from mono_agent.service.mcp_server import create_server
from mono_agent.tools import TOOLS, run_tool


def test_mcp_lists_and_calls_the_same_tools(ctx):
    args = {"date_from": "2025-10-01", "date_to": "2025-10-31", "group_by": "category"}

    async def session():
        async with Client(create_server(ctx)) as client:
            listed = await client.list_tools()
            ok = await client.call_tool("aggregate_spending", args)
            bad = await client.call_tool("aggregate_spending", {"category": "jars"})
            return listed.tools, ok, bad

    tools, ok, bad = asyncio.run(session())
    assert {t.name for t in tools} == set(TOOLS)
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)
    assert not ok.is_error
    assert ok.structured_content == run_tool(ctx, "aggregate_spending", args)[0]
    # Validation errors go back to the client's model as data, like inside the graph.
    assert bad.is_error and bad.structured_content["error"] == "invalid arguments"
