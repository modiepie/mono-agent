"""Tool outputs checked against ground truth computed directly from the generator."""

from datetime import datetime

from mono_agent.guardrails import REMOVED
from mono_agent.synthetic.generator import KYIV, generate
from mono_agent.tools import TOOLS, run_tool
from mono_agent.tools.categories import NON_SPENDING, category

DATA = generate(42)
MAIN = DATA.statements[DATA.main_account]
OCT = {"date_from": "2025-10-01", "date_to": "2025-10-31"}
SEP = {"date_from": "2025-09-01", "date_to": "2025-09-30"}


def in_month(item, month: int) -> bool:
    return datetime.fromtimestamp(item.time, KYIV).month == month


def truth_spent(month: int, cat: str | None = None) -> float:
    return round(
        -sum(
            i.amount
            for i in MAIN
            if in_month(i, month)
            and category(i.mcc, i.description) not in NON_SPENDING
            and (cat is None or category(i.mcc, i.description) == cat)
        )
        / 100,
        2,
    )


def test_aggregate_matches_ground_truth(ctx):
    out, ok = run_tool(ctx, "aggregate_spending", {**OCT, "group_by": "category", "top_n": 50})
    assert ok
    assert out["total_spent"] == truth_spent(10)
    taxi = next(g for g in out["groups"] if g["key"] == "taxi")
    assert taxi["total"] == truth_spent(10, "taxi")
    assert taxi["count"] == sum(1 for i in MAIN if in_month(i, 10) and i.mcc == 4121)


def test_aggregate_by_month_is_chronological(ctx):
    out, _ = run_tool(
        ctx,
        "aggregate_spending",
        {"date_from": "2025-01-01", "date_to": "2025-12-31", "group_by": "month"},
    )
    assert [g["key"] for g in out["groups"]][:3] == ["2025-01", "2025-02", "2025-03"]


def test_compare_explains_taxi_spike(ctx):
    out, ok = run_tool(ctx, "compare_periods", {"period_a": SEP, "period_b": OCT})
    assert ok
    groups = {g["key"]: g for g in out["groups"]}
    assert groups["taxi"]["count_b"] > 2 * groups["taxi"]["count_a"]
    assert groups["car"]["total_a"] == 0 and groups["car"]["total_b"] == 23_050
    assert groups["fuel"]["delta"] < 0
    assert out["total_b"] == truth_spent(10) and out["total_a"] == truth_spent(9)
    assert out["period_a"] == "2025-09-01..2025-09-30"


def test_refunds_reduce_spending(ctx):
    out, _ = run_tool(
        ctx,
        "query_transactions",
        {"date_from": "2025-03-01", "date_to": "2025-03-31", "text": "rozetka"},
    )
    assert out["total_count"] == 3  # two charges + one refund
    assert out["total_amount"] == -3_499


def test_subscriptions_find_planted_signals(ctx):
    out, ok = run_tool(
        ctx, "detect_subscriptions", {"date_from": "2025-01-01", "date_to": "2025-12-31"}
    )
    assert ok
    subs = {s["merchant"]: s for s in out["subscriptions"]}
    assert {"Spotify", "Netflix.com", "MEGOGO", "Sport Life", "Google YouTube Premium"} <= set(subs)
    assert "Барбершоп Frisor" not in subs  # every two months: not a monthly subscription
    assert any("price changed" in s for s in subs["Netflix.com"]["signals"])
    assert any("no charge since" in s for s in subs["MEGOGO"]["signals"])
    assert any("Spotify" in s for s in subs["Google YouTube Premium"]["signals"])
    assert "Duolingo" in {c["description"] for c in out["irregular_digital_charges"]}


def test_forecast_uses_last_full_months(ctx):
    out, ok = run_tool(
        ctx, "forecast_savings", {"category": "restaurants", "cut_percent": 30, "months": 3}
    )
    assert ok
    baseline = round(
        (
            truth_spent(10, "restaurants")
            + truth_spent(11, "restaurants")
            + truth_spent(12, "restaurants")
        )
        / 3,
        2,
    )
    assert out["baseline_period"] == "2025-10 to 2025-12"
    assert out["baseline_monthly_spend"] == baseline
    assert out["total_saving"] == round(round(baseline * 0.3, 2) * 3, 2)


def test_injection_text_never_leaves_the_tool(ctx):
    out, _ = run_tool(
        ctx,
        "query_transactions",
        {
            "date_from": "2025-01-01",
            "date_to": "2025-12-31",
            "limit": 50,
            "min_amount": 250,
            "max_amount": 250,
        },
    )
    dump = str(out)
    assert "IGNORE ALL PREVIOUS" not in dump
    assert REMOVED in dump


def test_sql_injection_in_merchant_is_just_a_string(ctx):
    out, ok = run_tool(ctx, "query_transactions", {**OCT, "text": "%' OR 1=1 --"})
    assert ok and out["total_count"] == 0
    out, ok = run_tool(ctx, "query_transactions", {**OCT, "text": "%"})
    assert ok and out["total_count"] == 0  # % is escaped, not a wildcard


def test_invalid_args_come_back_as_errors(ctx):
    out, ok = run_tool(
        ctx, "query_transactions", {"date_from": "2025-10-31", "date_to": "2025-10-01"}
    )
    assert not ok and out["error"] == "invalid arguments"
    out, ok = run_tool(ctx, "aggregate_spending", {**OCT, "group_by": "merchant; DROP TABLE"})
    assert not ok
    out, ok = run_tool(ctx, "drop_tables", {})
    assert not ok


def test_every_tool_schema_is_valid_json_schema():
    for spec in TOOLS.values():
        schema = spec.schema()
        assert schema["function"]["name"] == spec.name
        assert "$ref" not in str(schema)  # refs inlined: small local models handle them poorly


def test_aggregate_reports_cashback_and_rejects_non_spending_category(ctx):
    out, ok = run_tool(ctx, "aggregate_spending", {**OCT, "group_by": "category", "top_n": 50})
    taxi = next(g for g in out["groups"] if g["key"] == "taxi")
    expected = sum(i.cashback_amount for i in MAIN if in_month(i, 10) and i.mcc == 4121) / 100
    assert taxi["cashback"] == round(expected, 2) > 0
    assert out["total_cashback"] >= taxi["cashback"]
    out, ok = run_tool(
        ctx, "aggregate_spending", {**OCT, "group_by": "month", "category": "cashback"}
    )
    assert not ok and "query_transactions" in str(out)


def test_empty_period_is_an_error_that_names_available_data(ctx):
    run_tool(ctx, "aggregate_spending", {**OCT, "group_by": "category"})  # cache October 2025
    out, ok = run_tool(
        ctx,
        "aggregate_spending",
        {"date_from": "2024-10-01", "date_to": "2024-10-31", "group_by": "category"},
    )
    assert not ok
    assert "2025-10-01" in out["error"] and "check the year" in out["error"]


def test_text_search_covers_comments_and_finds_rent(ctx):
    out, ok = run_tool(
        ctx,
        "query_transactions",
        {"date_from": "2025-01-01", "date_to": "2025-12-31", "text": "оренда"},
    )
    assert ok and out["total_count"] == 12 and out["total_amount"] == -180_000


def test_refunds_kind_lists_only_returned_purchases(ctx):
    out, ok = run_tool(
        ctx,
        "query_transactions",
        {"date_from": "2025-01-01", "date_to": "2025-12-31", "kind": "refunds"},
    )
    assert ok and sorted(i["amount"] for i in out["items"]) == [3200, 3499]


def test_contradictory_filters_are_an_error_not_an_empty_result(ctx):
    args = {"date_from": "2025-01-01", "date_to": "2025-12-31", "category": "jars"}
    out, ok = run_tool(ctx, "query_transactions", {**args, "kind": "spending"})
    assert not ok and "kind=all" in str(out)
    out, ok = run_tool(ctx, "query_transactions", {**args, "kind": "all"})
    assert ok and out["total_amount"] == -57_000
