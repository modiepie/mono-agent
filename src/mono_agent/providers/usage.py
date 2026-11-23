"""Per-request token, cost and latency tracking as a LangChain callback.

Attach one tracker per request: `graph.invoke(..., {"callbacks": [tracker]})`. Every chat
model call inside the graph reports its token usage here, keyed by provider:model.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

# USD per 1M tokens (input, output). Local models cost nothing per token.
Prices = dict[str, tuple[float, float]]


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


@dataclass(eq=False)  # callbacks are kept in sets: must stay hashable by identity
class UsageTracker(BaseCallbackHandler):
    by_model: dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))
    _started: dict[UUID, tuple[str, float]] = field(default_factory=dict)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        name = f"{meta.get('ls_provider', '?')}:{meta.get('ls_model_name', '?')}"
        self._started[run_id] = (name, time.perf_counter())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        name, start = self._started.pop(run_id, ("?:?", time.perf_counter()))
        usage = self.by_model[name]
        usage.calls += 1
        usage.seconds += time.perf_counter() - start
        for gens in response.generations:
            for gen in gens:
                meta = (
                    getattr(gen.message, "usage_metadata", None)
                    if isinstance(gen, ChatGeneration)
                    else None
                )
                if meta:
                    usage.input_tokens += meta.get("input_tokens", 0)
                    usage.output_tokens += meta.get("output_tokens", 0)

    def totals(self) -> dict[str, int]:
        return {
            "calls": sum(u.calls for u in self.by_model.values()),
            "input_tokens": sum(u.input_tokens for u in self.by_model.values()),
            "output_tokens": sum(u.output_tokens for u in self.by_model.values()),
        }

    def cost_usd(self, prices: Prices) -> float | None:
        """None if a used model has no known price (better than reporting a wrong zero)."""
        total = 0.0
        for name, u in self.by_model.items():
            model = name.split(":", 1)[1]
            price = prices.get(name) or prices.get(model)
            if name.startswith("ollama:"):
                price = price or (0.0, 0.0)
            if price is None:
                return None
            total += (u.input_tokens * price[0] + u.output_tokens * price[1]) / 1_000_000
        return round(total, 6)
