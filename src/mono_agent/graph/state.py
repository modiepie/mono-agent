"""Graph state. `messages` is the conversation kept across turns (checkpointed); every other
field belongs to the current turn and is reset by the guardrail node at the start of a turn."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Step = Literal["transactions", "knowledge", "export"]


class RouterDecision(BaseModel):
    """How to handle the user's latest message."""

    intent: Literal["transactions", "knowledge", "mixed", "export", "out_of_scope"] = Field(
        description="transactions: about the user's own spending/income/subscriptions/savings. "
        "knowledge: about monobank products, tariffs, terms. mixed: needs both. "
        "export: user wants a report file. out_of_scope: anything else."
    )
    steps: list[Step] = Field(description="Ordered steps to run, e.g. ['transactions', 'export']")


class ToolResult(TypedDict):
    id: str
    tool: str
    args: dict
    output: dict
    ok: bool


class Citation(TypedDict):
    kind: Literal["tool", "doc"]
    ref: str


class InputState(TypedDict):
    """What a caller sends in: just the new message(s)."""

    messages: Annotated[list[AnyMessage], add_messages]


class AgentState(TypedDict):
    # All keys are always present after the guardrail node, which resets them every turn.
    messages: Annotated[list[AnyMessage], add_messages]
    # --- per turn ---
    blocked: Literal["injection", "out_of_scope"] | None
    intent: str
    plan: list[Step]  # remaining steps; each step node pops itself when done
    scratchpad: list[AnyMessage]  # ReAct tool-calling messages, not kept in the conversation
    agent_steps: int
    tool_results: list[ToolResult]
    retrieved_docs: list[dict]  # filled by the knowledge agent (milestone 3)
    citations: list[Citation]
    export_path: str | None
