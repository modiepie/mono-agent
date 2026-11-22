from datetime import date

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from mono_agent.config import Settings
from mono_agent.mono.cache import Cache
from mono_agent.mono.factory import build_client
from mono_agent.tools.finance import ToolContext


@pytest.fixture
def ctx() -> ToolContext:
    client = build_client(Settings(_env_file=None, mono_token=None))  # type: ignore[call-arg]
    return ToolContext(client, Cache(":memory:"), date(2025, 12, 31))


class Scripted(BaseChatModel):
    """Fake chat model: returns the scripted replies in order and records every prompt."""

    replies: list[AIMessage] = Field(default_factory=list)
    prompts: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(list(messages))
        if not self.replies:
            raise AssertionError("model called more times than scripted")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def route(intent: str, steps: list[str]) -> AIMessage:
    return tool_call("RouterDecision", {"intent": intent, "steps": steps}, "r")
