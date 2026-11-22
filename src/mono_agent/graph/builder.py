"""LangGraph assembly.

    START -> guardrail -> router -> [transactions_agent <-> transactions_tools]
                                  -> knowledge -> export (interrupt) -> composer -> END
    guardrail / router can short-cut to `refuse`.

The plan from the router is a list of steps; every step node pops itself when done, and
`_next_step` sends the graph to the next one (or to the composer when the list is empty).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, interrupt

from mono_agent.graph import prompts
from mono_agent.graph.state import AgentState, InputState, RouterDecision, Step, ToolResult
from mono_agent.guardrails import looks_like_injection, numbers_in, ungrounded_numbers
from mono_agent.tools import TOOLS, run_tool
from mono_agent.tools.finance import ToolContext

MAX_AGENT_STEPS = 8  # tool-calling rounds per turn; stops runaway ReAct loops
HISTORY = 6  # previous messages shown to router/agent: enough for follow-ups, cheap


@dataclass
class Models:
    router: BaseChatModel
    agent: BaseChatModel
    composer: BaseChatModel


def lang(text: str) -> str:
    return "uk" if re.search(r"[а-яіїєґ]", text, re.IGNORECASE) else "en"


def _question(state: AgentState) -> str:
    return next(str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def _history(state: AgentState) -> list[AnyMessage]:
    convo: list[AnyMessage] = [
        m for m in state["messages"] if isinstance(m, HumanMessage | AIMessage)
    ]
    return convo[-HISTORY:]


class Nodes:
    def __init__(self, models: Models, ctx: ToolContext, export_dir: Path) -> None:
        self.models = models
        self.ctx = ctx
        self.export_dir = export_dir
        self.agent_llm = models.agent.bind_tools([t.schema() for t in TOOLS.values()])
        self.router_llm = models.router.with_structured_output(RouterDecision)

    def guardrail(self, state: AgentState) -> dict:
        """Start of a turn: reset per-turn state, block injection attempts in the user input."""
        blocked = "injection" if looks_like_injection(_question(state)) else None
        return {
            "blocked": blocked,
            "intent": "",
            "plan": [],
            "scratchpad": [],
            "agent_steps": 0,
            "tool_results": [],
            "retrieved_docs": [],
            "citations": [],
            "export_path": None,
        }

    def router(self, state: AgentState) -> dict:
        system = SystemMessage(prompts.ROUTER.format(today=self.ctx.today))
        decision = cast(RouterDecision, self.router_llm.invoke([system, *_history(state)]))
        steps = list(dict.fromkeys(decision.steps))  # drop duplicates, keep order
        if decision.intent == "out_of_scope":
            return {"intent": decision.intent, "blocked": "out_of_scope", "plan": []}
        if decision.intent == "mixed":
            steps = steps or ["transactions", "knowledge"]
        if not steps:  # model gave an intent but no steps: derive the obvious plan
            steps = {"export": ["transactions", "export"]}.get(decision.intent, [decision.intent])
        return {"intent": decision.intent, "plan": steps}

    def transactions_agent(self, state: AgentState) -> dict:
        steps = state.get("agent_steps", 0)
        if steps >= MAX_AGENT_STEPS:
            return {"plan": state["plan"][1:]}
        system = SystemMessage(prompts.TRANSACTIONS_AGENT.format(today=self.ctx.today))
        reply = self.agent_llm.invoke([system, *_history(state), *state.get("scratchpad", [])])
        update: dict = {
            "scratchpad": [*state.get("scratchpad", []), reply],
            "agent_steps": steps + 1,
        }
        if not getattr(reply, "tool_calls", None):
            update["plan"] = state["plan"][1:]
        return update

    def transactions_tools(self, state: AgentState) -> dict:
        last = cast(AIMessage, state["scratchpad"][-1])
        messages, results = [], list(state.get("tool_results", []))
        for call in last.tool_calls:
            output, ok = run_tool(self.ctx, call["name"], call["args"])
            call_id = call["id"] or f"call_{len(results)}"
            results.append(
                ToolResult(id=call_id, tool=call["name"], args=call["args"], output=output, ok=ok)
            )
            messages.append(
                ToolMessage(
                    json.dumps(output, ensure_ascii=False),
                    tool_call_id=call_id,
                    status="success" if ok else "error",
                )
            )
        return {"scratchpad": [*state["scratchpad"], *messages], "tool_results": results}

    def knowledge(self, state: AgentState) -> dict:
        # Milestone 3 replaces this with hybrid retrieval over data/docs.
        return {"retrieved_docs": [], "plan": state["plan"][1:]}

    def export(self, state: AgentState) -> dict:
        ok = [r for r in state.get("tool_results", []) if r["ok"]]
        approved = interrupt(
            {"action": "export_report", "question": _question(state), "tool_results": len(ok)}
        )
        update: dict = {"plan": state["plan"][1:]}
        if approved is True:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            path = self.export_dir / f"report-{datetime.now():%Y%m%d-%H%M%S}.md"
            body = [f"# Report\n\n**Question:** {_question(state)}\n"]
            for r in ok:
                body.append(
                    f"## {r['tool']}\n\nArguments: `{json.dumps(r['args'])}`\n\n"
                    f"```json\n{json.dumps(r['output'], ensure_ascii=False, indent=2)}\n```\n"
                )
            path.write_text("\n".join(body), encoding="utf-8")
            update["export_path"] = str(path)
        return update

    def composer(self, state: AgentState) -> dict:
        question = _question(state)
        results = [r for r in state.get("tool_results", []) if r["ok"]]
        docs = state.get("retrieved_docs", [])
        if docs:
            docs_text = json.dumps(docs, ensure_ascii=False)
        elif state.get("intent") in ("knowledge", "mixed"):
            docs_text = prompts.NO_KNOWLEDGE_YET
        else:
            docs_text = "none"
        compact = [
            {"id": r["id"], "tool": r["tool"], "args": r["args"], "output": r["output"]}
            for r in results
        ]
        system = SystemMessage(
            prompts.COMPOSER.format(
                today=self.ctx.today,
                tool_results=json.dumps(compact, ensure_ascii=False),
                documents=docs_text,
            )
        )
        convo = [system, *_history(state)]
        answer = str(self.models.composer.invoke(convo).content)

        # Every number in the answer must come from tool results, their arguments, the
        # question or today's date. One retry with feedback, then a visible warning.
        allowed = numbers_in([compact, docs, question, str(self.ctx.today)])
        if bad := ungrounded_numbers(answer, allowed):
            fix = HumanMessage(prompts.FIX_NUMBERS.format(numbers=", ".join(map(str, bad))))
            answer = str(self.models.composer.invoke([*convo, AIMessage(answer), fix]).content)
            if ungrounded_numbers(answer, allowed):
                answer += (
                    "\n\n⚠️ Деякі числа не вдалося перевірити."
                    if lang(question) == "uk"
                    else "\n\n⚠️ Some numbers could not be verified."
                )
        if state.get("export_path"):
            answer += f"\n\n📄 {state['export_path']}"
        if any(r["tool"] == "forecast_savings" for r in results):
            answer += "\n\n" + prompts.DISCLAIMER[lang(question)]
        citations = [{"kind": "tool", "ref": r["id"]} for r in results] + [
            {"kind": "doc", "ref": d.get("id", "")} for d in docs
        ]
        return {"messages": [AIMessage(answer)], "citations": citations}

    def refuse(self, state: AgentState) -> dict:
        reason = state.get("blocked") or "out_of_scope"
        return {"messages": [AIMessage(prompts.REFUSALS[reason][lang(_question(state))])]}


def _next_step(state: AgentState) -> str:
    plan: list[Step] = state.get("plan", [])
    return plan[0] if plan else "composer"


def _after_agent(state: AgentState) -> str:
    # The agent pops "transactions" from the plan when it is done (or out of steps); while it
    # is still first in the plan, pending tool calls must run.
    last = state["scratchpad"][-1] if state.get("scratchpad") else None
    still_working = state.get("plan", [])[:1] == ["transactions"]
    if still_working and last is not None and getattr(last, "tool_calls", None):
        return "transactions_tools"
    return _next_step(state)


def build_graph(
    models: Models, ctx: ToolContext, export_dir: Path, checkpointer: Checkpointer = None
):
    nodes = Nodes(models, ctx, export_dir)
    g = StateGraph(AgentState, input_schema=InputState)
    g.add_node("guardrail", nodes.guardrail)
    g.add_node("router", nodes.router)
    g.add_node("transactions", nodes.transactions_agent)
    g.add_node("transactions_tools", nodes.transactions_tools)
    g.add_node("knowledge", nodes.knowledge)
    g.add_node("export", nodes.export)
    g.add_node("composer", nodes.composer)
    g.add_node("refuse", nodes.refuse)

    steps = ["transactions", "knowledge", "export", "composer"]
    g.add_edge(START, "guardrail")
    g.add_conditional_edges(
        "guardrail", lambda s: "refuse" if s.get("blocked") else "router", ["refuse", "router"]
    )
    g.add_conditional_edges(
        "router", lambda s: "refuse" if s.get("blocked") else _next_step(s), ["refuse", *steps]
    )
    g.add_conditional_edges("transactions", _after_agent, ["transactions_tools", *steps])
    g.add_edge("transactions_tools", "transactions")
    g.add_conditional_edges("knowledge", _next_step, steps)
    g.add_conditional_edges("export", _next_step, steps)
    g.add_edge("composer", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer)
