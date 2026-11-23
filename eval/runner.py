"""Run the graph on every question for one provider config and score the results."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from eval import metrics
from eval.dataset import Question
from eval.judge import judge
from eval.recording import Recording
from mono_agent.app import build_app
from mono_agent.config import Settings
from mono_agent.graph.builder import Models
from mono_agent.providers import chat_model, missing_key
from mono_agent.providers.usage import UsageTracker
from mono_agent.rag import build_retriever, retrieve_fn


def _log(*args: object) -> None:
    print(*args, flush=True)  # visible live even when stdout is piped


CONFIGS = Path(__file__).parent / "configs"


@dataclass
class EvalConfig:
    name: str
    router: str
    agent: str
    composer: str
    judge: str
    description: str = ""
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str) -> "EvalConfig":
        data = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text("utf-8"))
        data["prices"] = {k: tuple(v) for k, v in (data.get("prices") or {}).items()}
        return cls(**data)

    def missing_keys(self) -> list[str]:
        specs = (self.router, self.agent, self.composer, self.judge)
        return sorted({k for s in specs if (k := missing_key(s))})


def eval_settings() -> Settings:
    # Always synthetic, whatever .env says: real bank data must never reach eval results.
    return Settings(_env_file=None, mono_token=None)  # type: ignore[call-arg]


def run_config(
    cfg: EvalConfig,
    questions: list[Question],
    recording: Recording | None = None,
    use_judge: bool = True,
    on_result=None,
    log=_log,
) -> list[dict]:
    settings = eval_settings()
    cache = recording  # None -> live calls
    models = Models(*(chat_model(s, cache=cache) for s in (cfg.router, cfg.agent, cfg.composer)))
    judge_model = chat_model(cfg.judge, cache=cache) if use_judge else None

    retrieve = None
    if recording is None or not recording.replay:
        retriever = build_retriever(settings)
        retrieve = retrieve_fn(retriever, settings) if retriever else None
    if recording is not None:
        retrieve = recording.wrap_retrieve(retrieve)

    results = []
    with TemporaryDirectory() as exports:
        app = build_app(
            settings,
            models=models,
            checkpointer=InMemorySaver(),
            retrieve=retrieve,
            export_dir=Path(exports),
            cache_path=":memory:",
        )
        for i, q in enumerate(questions, 1):
            r = run_question(app, q, judge_model, cfg.prices)
            results.append(r)
            if on_result:  # e.g. save progress, so an interrupted run can be resumed
                on_result(results)
            log(
                f"[{i}/{len(questions)}] {q.id}: {'PASS' if r['pass'] else 'fail'} "
                f"({r['latency_s']:.1f}s){' ' + r['error'] if r['error'] else ''}"
            )
    return results


def run_question(app, q: Question, judge_model, prices) -> dict:
    tracker = UsageTracker()
    config: RunnableConfig = {"configurable": {"thread_id": f"eval-{q.id}"}, "callbacks": [tracker]}
    start = time.perf_counter()
    state, error = {}, None
    try:
        state = app.invoke({"messages": [HumanMessage(q.q)]}, config)
        while state.get("__interrupt__"):  # export: approve, as a user would
            state = app.invoke(Command(resume=True), config)
    except Exception as exc:  # one broken question must not stop the whole run
        error = f"{type(exc).__name__}: {exc}"[:300]
    latency = time.perf_counter() - start

    messages = state.get("messages", [])
    answer = str(messages[-1].content) if messages and not error else ""
    calls = state.get("tool_results", [])
    docs = state.get("retrieved_docs", [])
    expected = q.expected_numbers()

    r: dict = {
        "id": q.id,
        "question": q.q,
        "tags": q.tags,
        "answer": answer,
        "error": error,
        "intent": state.get("intent", ""),
        "blocked": state.get("blocked"),
        "tools": [{"tool": c["tool"], "args": c["args"], "ok": c["ok"]} for c in calls],
        "docs": [d["id"] for d in docs],
        "latency_s": round(latency, 2),
        **tracker.totals(),
        "cost_usd": tracker.cost_usd(prices),
        "routing_ok": metrics.routing_ok(q, state.get("intent", ""), state.get("blocked")),
        "numbers": metrics.numbers_score(expected, answer),
        "expected_numbers": expected,
        "safety_ok": metrics.safety_ok(q, answer),
        "retrieval": metrics.retrieval_scores(q.evidence, docs, k=5),
    }
    scores = metrics.tool_scores(q.tools, calls)
    r["tool_names"], r["tool_args"] = scores if scores else (None, None)

    r["judge"] = None
    if judge_model is not None and answer and not q.refuse:
        context = {
            "tool_results": [
                {"tool": c["tool"], "args": c["args"], "output": c["output"]}
                for c in calls
                if c["ok"]
            ],
            "documents": [
                {"id": d["id"], "section": d["section"], "text": d["text"]} for d in docs
            ],
        }
        try:
            v = judge(judge_model, q.q, context, q.reference, answer)
            r["judge"] = v.model_dump()
        except Exception as exc:
            r["judge"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

    r["pass"] = passed(r)
    return r


def rescore(r: dict, q: Question) -> dict:
    """Recompute the deterministic metrics of a stored answer against the current dataset.
    Used after fixing an expectation, so old runs are judged by the same rules as new ones.
    Retrieval and judge scores are kept: they don't depend on the dataset's expectations."""
    expected = q.expected_numbers()
    r = {
        **r,
        "tags": q.tags,
        "expected_numbers": expected,
        "routing_ok": metrics.routing_ok(q, r["intent"], r["blocked"]),
        "numbers": metrics.numbers_score(expected, r["answer"]),
        "safety_ok": metrics.safety_ok(q, r["answer"]),
    }
    scores = metrics.tool_scores(q.tools, r["tools"])
    r["tool_names"], r["tool_args"] = scores if scores else (None, None)
    r["pass"] = passed(r)
    return r


def passed(r: dict) -> bool:
    """Strict: every applicable deterministic check must be perfect."""
    checks = [r["routing_ok"], r["error"] is None]
    checks += [r[k] == 1 for k in ("tool_args", "numbers") if r[k] is not None]
    checks += [r["safety_ok"]] if r["safety_ok"] is not None else []
    checks += [r["retrieval"]["recall"] == 1] if r["retrieval"] else []
    return all(checks)


def save(results: list[dict], cfg: EvalConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg.__dict__,
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "questions": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
