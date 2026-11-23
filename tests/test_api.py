"""The real FastAPI app on a graph with scripted models."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from mono_agent.config import Settings
from mono_agent.graph.builder import Models, build_graph
from mono_agent.service.api import create_api
from tests.conftest import Scripted, route, tool_call

OCT_TAXI = {
    "date_from": "2025-10-01",
    "date_to": "2025-10-31",
    "group_by": "category",
    "category": "taxi",
}


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def client_for(ctx, tmp_path: Path, router=(), agent=(), composer=()) -> TestClient:
    models = Models(
        Scripted(replies=list(router)),
        Scripted(replies=list(agent)),
        Scripted(replies=list(composer)),
    )
    settings = Settings(_env_file=None, mono_token=None)  # type: ignore[call-arg]
    app = create_api(lambda: build_graph(models, ctx, tmp_path, InMemorySaver()), settings)
    return TestClient(app)


def test_health(ctx, tmp_path):
    with client_for(ctx, tmp_path) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["data"] == "synthetic"


def test_chat_streams_progress_then_verified_answer(ctx, tmp_path):
    with client_for(
        ctx,
        tmp_path,
        router=[route("transactions", ["transactions"])],
        agent=[tool_call("aggregate_spending", OCT_TAXI), AIMessage("found")],
        composer=[AIMessage("5 864 грн на таксі.")],
    ) as client:
        evs = events(client.post("/chat", json={"message": "Таксі в жовтні?"}))
    names = [e for e, _ in evs]
    assert names[0] == "thread" and names[-1] == "answer"
    steps = [d for e, d in evs if e == "step"]
    assert [s["node"] for s in steps][:2] == ["guardrail", "router"]
    tool_step = next(s for s in steps if s["node"] == "transactions_tools")
    assert tool_step["detail"]["tool"] == "aggregate_spending"
    answer = evs[-1][1]
    assert answer["text"] == "5 864 грн на таксі."
    assert answer["tools"][0]["tool"] == "aggregate_spending"


def test_conversation_continues_on_same_thread(ctx, tmp_path):
    with client_for(
        ctx,
        tmp_path,
        router=[route("out_of_scope", [])] * 2,
    ) as client:
        first = events(client.post("/chat", json={"message": "poem?"}))
        thread = first[0][1]["thread_id"]
        second = events(client.post("/chat", json={"message": "joke?", "thread_id": thread}))
    assert second[0][1]["thread_id"] == thread


def test_export_pauses_for_confirmation_and_resumes(ctx, tmp_path):
    with client_for(
        ctx,
        tmp_path,
        router=[route("export", ["transactions", "export"])],
        agent=[tool_call("aggregate_spending", OCT_TAXI), AIMessage("done")],
        composer=[AIMessage("Звіт: 5 864 грн.")],
    ) as client:
        paused = events(client.post("/chat", json={"message": "Експортуй звіт за жовтень"}))
        assert paused[-1][0] == "confirm"
        thread = paused[0][1]["thread_id"]
        done = events(client.post(f"/chat/{thread}/resume", json={"approve": True}))
        again = client.post(f"/chat/{thread}/resume", json={"approve": True})
    assert done[-1][0] == "answer" and done[-1][1]["export_path"]
    assert again.status_code == 409  # nothing left to confirm


@pytest.mark.parametrize(
    "body", [{"message": ""}, {"message": "x" * 2001}, {"message": "hi", "thread_id": "../etc"}]
)
def test_invalid_input_is_rejected(ctx, tmp_path, body):
    with client_for(ctx, tmp_path) as client:
        assert client.post("/chat", json=body).status_code == 422


def test_model_failure_becomes_an_error_event_not_a_crash(ctx, tmp_path):
    with client_for(ctx, tmp_path) as client:  # nothing scripted: the router call fails
        evs = events(client.post("/chat", json={"message": "taxi?"}))
    assert evs[-1] == ("error", {"message": "internal error"})


def test_abandoned_stream_does_not_block_the_next_request(ctx, tmp_path):
    """Regression: a client that disconnected mid-stream (Streamlit reruns do this) used to
    keep the run lock forever, and every later request hung."""
    with client_for(
        ctx,
        tmp_path,
        router=[route("out_of_scope", [])] * 2,
    ) as client:
        with client.stream("POST", "/chat", json={"message": "poem?"}) as response:
            next(response.iter_lines())  # read one line, then walk away
        evs = events(client.post("/chat", json={"message": "another poem?"}))
    assert evs[-1][0] == "answer"
