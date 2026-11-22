"""Tool registry for the transactions agent: name, description, input model, function."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ValidationError

from mono_agent.tools import finance
from mono_agent.tools.finance import ToolContext
from mono_agent.tools.schemas import (
    AggregateInput,
    CompareInput,
    ForecastInput,
    NoInput,
    QueryInput,
    SubscriptionsInput,
    SyncInput,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], BaseModel]

    def schema(self) -> dict:
        """OpenAI-style tool schema (refs inlined) that every LangChain chat model accepts."""
        tool = convert_to_openai_tool(self.input_model)
        tool["function"]["name"] = self.name
        tool["function"]["description"] = self.description
        return tool


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "sync_statement",
            "Refresh cached transactions from the bank for a period. "
            "Other tools sync automatically; use only when asked to refresh.",
            SyncInput,
            finance.sync_statement,
        ),
        ToolSpec(
            "query_transactions",
            "List individual transactions with filters "
            "(category, merchant substring, spending/income, amount range).",
            QueryInput,
            finance.query_transactions,
        ),
        ToolSpec(
            "aggregate_spending",
            "Total spending (net of refunds, transfers excluded) "
            "grouped by category, merchant, month, week or day.",
            AggregateInput,
            finance.aggregate_spending,
        ),
        ToolSpec(
            "compare_periods",
            "Compare spending between two periods per category or "
            "merchant, with counts and average ticket to explain why it changed.",
            CompareInput,
            finance.compare_periods,
        ),
        ToolSpec(
            "detect_subscriptions",
            "Find recurring monthly charges and flag price "
            "changes, cancellations, overlapping services and irregular digital charges.",
            SubscriptionsInput,
            finance.detect_subscriptions,
        ),
        ToolSpec(
            "currency_rates",
            "Current monobank exchange rates for USD, EUR, PLN.",
            NoInput,
            finance.currency_rates,
        ),
        ToolSpec(
            "forecast_savings",
            "How much is saved by cutting a category by N% for M "
            "months, using the average of recent full months as the baseline.",
            ForecastInput,
            finance.forecast_savings,
        ),
    ]
}


def run_tool(ctx: ToolContext, name: str, args: dict) -> tuple[dict, bool]:
    """Validate args, run the tool. Returns (JSON-able output, ok). Errors go back to the
    model as data so it can correct its call instead of crashing the graph."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"unknown tool {name!r}"}, False
    try:
        parsed = spec.input_model.model_validate(args)
    except ValidationError as exc:
        return {
            "error": "invalid arguments",
            "details": json.loads(exc.json(include_url=False)),
        }, False
    try:
        return spec.fn(ctx, parsed).model_dump(mode="json"), True
    except ValueError as exc:
        return {"error": str(exc)}, False
