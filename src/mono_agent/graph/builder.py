"""LangGraph assembly.

    START -> guardrail -> router -> [transactions_agent <-> transactions_tools]
                                  -> knowledge -> export (interrupt) -> composer -> END
    guardrail / router can short-cut to `refuse`.

The plan from the router is a list of steps; every step node pops itself when done, and
`_next_step` sends the graph to the next one (or to the composer when the list is empty).
"""

import json
import re
from collections.abc import Callable
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
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, interrupt

from mono_agent.graph import prompts
from mono_agent.graph.state import AgentState, InputState, RouterDecision, Step, ToolResult
from mono_agent.guardrails import looks_like_injection, numbers_in, ungrounded_numbers
from mono_agent.tools import TOOLS, run_tool
from mono_agent.tools.finance import ToolContext

MAX_AGENT_STEPS = 8  # tool-calling rounds per turn; stops runaway ReAct loops
HISTORY = 6  # previous messages shown to router/agent: enough for follow-ups, cheap
# Models sometimes cite tool calls ("[detect_subscriptions]", "[07607e32-...]"): internal
# noise for a user. Document citations ("[credit-limit#2]") are kept.
_INTERNAL_REFS = re.compile(
    r"\s?\[(?:" + "|".join(TOOLS) + r"|[0-9a-f]{8}-[0-9a-f-]{27,}|call_\w+)\]"
)
_CITE = re.compile(r"\[([\w.-]+#\d+)\]")

# query -> retrieved chunks as dicts (id, doc_id, title, section, date, sample, text, score)
Retrieve = Callable[[str], list[dict]]


@dataclass
class Models:
    router: BaseChatModel
    agent: BaseChatModel
    composer: BaseChatModel
    fallback: BaseChatModel | None = None  # used by every node when its own model fails


def _with_fallback(primary: Runnable, backup: Runnable | None) -> Runnable:
    """Retry the same call on the backup model if the primary raises (timeout, 429, 5xx)."""
    return primary.with_fallbacks([backup]) if backup is not None else primary


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
    def __init__(
        self, models: Models, ctx: ToolContext, export_dir: Path, retrieve: Retrieve | None = None
    ) -> None:
        self.models = models
        self.retrieve = retrieve
        self.ctx = ctx
        self.export_dir = export_dir
        # Each use is bound (tools, schema) before adding the fallback: a fallback must be
        # bound the same way, or it would answer without tools.
        schemas = [t.schema() for t in TOOLS.values()]
        fb = models.fallback
        self.agent_llm = _with_fallback(
            models.agent.bind_tools(schemas), fb.bind_tools(schemas) if fb else None
        )
        self.router_llm = _with_fallback(
            models.router.with_structured_output(RouterDecision),
            fb.with_structured_output(RouterDecision) if fb else None,
        )
        self.composer_llm = _with_fallback(models.composer, fb)
        self.rewrite_llm = _with_fallback(models.router, fb)

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
        """RAG step. A follow-up ("а в євро?") is first rewritten into a standalone query;
        a first message is searched as is, which saves a model call."""
        update: dict = {"plan": state["plan"][1:], "retrieved_docs": []}
        if self.retrieve is None:  # e.g. Qdrant not running: answer says no documents
            return update
        query = _question(state)
        if len(_history(state)) > 1:
            system = SystemMessage(prompts.STANDALONE_QUERY)
            query = str(self.rewrite_llm.invoke([system, *_history(state)]).content).strip()
        update["retrieved_docs"] = self.retrieve(query)
        return update

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
            shown = [{k: d[k] for k in ("id", "title", "section", "date", "text")} for d in docs]
            docs_text = json.dumps(shown, ensure_ascii=False)
        elif state.get("intent") in ("knowledge", "mixed"):
            docs_text = prompts.NO_KNOWLEDGE
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
        if state.get("intent") == "export":
            # Only on export turns, so every other prompt stays unchanged.
            note = prompts.EXPORT_DONE if state.get("export_path") else prompts.EXPORT_DECLINED
            system = SystemMessage(str(system.content) + "\n" + note)
        convo = [system, *_history(state)]
        answer = str(self.composer_llm.invoke(convo).content)

        # Every number in the answer must come from tool results, their arguments, the
        # question or today's date. One retry with feedback, then a visible warning.
        allowed = numbers_in([compact, docs, question, str(self.ctx.today)])
        if bad := ungrounded_numbers(answer, allowed):
            fix = HumanMessage(prompts.FIX_NUMBERS.format(numbers=", ".join(map(str, bad))))
            answer = str(self.composer_llm.invoke([*convo, AIMessage(answer), fix]).content)
            if ungrounded_numbers(answer, allowed):
                answer += (
                    "\n\n⚠️ Деякі числа не вдалося перевірити."
                    if lang(question) == "uk"
                    else "\n\n⚠️ Some numbers could not be verified."
                )
        answer = _INTERNAL_REFS.sub("", answer)
        if state.get("export_path"):
            answer += f"\n\n📄 {state['export_path']}"
        if any(r["tool"] == "forecast_savings" for r in results):
            answer += "\n\n" + prompts.DISCLAIMER[lang(question)]
        # Doc citations = chunks the answer actually cites (and that were really retrieved).
        retrieved = {d["id"] for d in docs}
        cited = [c for c in dict.fromkeys(_CITE.findall(answer)) if c in retrieved]
        citations = [{"kind": "tool", "ref": r["id"]} for r in results] + [
            {"kind": "doc", "ref": c} for c in cited
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
    models: Models,
    ctx: ToolContext,
    export_dir: Path,
    checkpointer: Checkpointer = None,
    retrieve: Retrieve | None = None,
):
    nodes = Nodes(models, ctx, export_dir, retrieve)
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
