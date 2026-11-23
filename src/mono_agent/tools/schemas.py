"""Pydantic input/output models for every tool. Inputs are what the LLM may send; anything
outside them is rejected before a query runs."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from mono_agent.tools.categories import NON_SPENDING, Category

Currency = Literal["UAH", "USD", "EUR"]


class Period(BaseModel):
    date_from: date = Field(description="First day, inclusive (YYYY-MM-DD)")
    date_to: date = Field(description="Last day, inclusive (YYYY-MM-DD)")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.date_to < self.date_from:
            raise ValueError("date_to is before date_from")
        if (self.date_to - self.date_from).days > 731:
            raise ValueError("period longer than 2 years")
        return self


class SyncInput(Period):
    currency: Currency = "UAH"


class QueryInput(Period):
    currency: Currency = Field(default="UAH", description="Which account (by currency)")
    category: Category | None = Field(
        default=None,
        description="transfers / jars / currency_exchange / cashback are money moves, not "
        "spending: use them with kind=all",
    )
    text: str | None = Field(
        default=None,
        max_length=60,
        description="Case-insensitive substring of the merchant/description or the transfer "
        "comment. Bank texts are mostly Ukrainian: search 'оренда', not 'rent'.",
    )
    kind: Literal["spending", "refunds", "income", "all"] = Field(
        default="all",
        description="spending = purchases (refunds included, net); refunds = money returned for "
        "purchases; income = salary, transfers in, cashback payouts; all = everything",
    )
    min_amount: float | None = Field(
        default=None, ge=0, description="Absolute amount, account currency"
    )
    max_amount: float | None = Field(default=None, ge=0)
    order_by: Literal["time", "amount"] = "time"
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def _consistent_filters(self) -> Self:
        # Seen in the eval: category=jars + kind=spending silently returned 0 rows and the
        # answer said "you put 0 UAH into jars". A contradiction is an error, not an empty set.
        if self.category in NON_SPENDING and self.kind in ("spending", "refunds"):
            raise ValueError(
                f"category={self.category} is not spending, so kind={self.kind} can never match;"
                " use kind=all"
            )
        return self


class AggregateInput(Period):
    group_by: Literal["category", "merchant", "month", "week", "day"]
    currency: Currency = "UAH"
    category: Category | None = Field(default=None, description="Only this category")
    top_n: int = Field(default=15, ge=1, le=50)

    @model_validator(mode="after")
    def _spending_category(self) -> Self:
        if self.category in NON_SPENDING:
            raise ValueError(
                f"{self.category} is not spending; use query_transactions with "
                f"category={self.category} (cashback earned per purchase is in the "
                "'cashback' field of aggregate_spending results)"
            )
        return self


class CompareInput(BaseModel):
    period_a: Period = Field(description="Earlier / baseline period (e.g. September)")
    period_b: Period = Field(description="Later period compared to the baseline (e.g. October)")
    group_by: Literal["category", "merchant"] = "category"
    category: Category | None = Field(default=None, description="Only this category")
    currency: Currency = "UAH"


class SubscriptionsInput(Period):
    pass


class ForecastInput(BaseModel):
    category: Category
    cut_percent: float = Field(gt=0, le=100)
    months: int = Field(ge=1, le=24)
    baseline_months: int = Field(
        default=3, ge=1, le=12, description="Full months used as the baseline"
    )


class NoInput(BaseModel):
    pass


# --- outputs -----------------------------------------------------------------------------


class SyncResult(BaseModel):
    fetched_from_api: int
    cached_transactions: int


class Transaction(BaseModel):
    time: str
    description: str | None
    category: str
    amount: float  # account currency, negative = money out
    currency: Currency
    operation_amount: float
    operation_currency: str
    cashback: float
    hold: bool
    comment: str | None = None
    suspicious_text_removed: bool = False


class TransactionList(BaseModel):
    total_count: int
    total_amount: float
    shown: int
    items: list[Transaction]


class Group(BaseModel):
    key: str
    total: float  # spending is positive
    count: int
    average: float
    cashback: float = 0  # cashback earned on these purchases


class AggregateResult(BaseModel):
    currency: Currency
    total_spent: float
    total_cashback: float
    groups: list[Group]


class GroupDelta(BaseModel):
    key: str
    total_a: float
    total_b: float
    delta: float
    delta_pct: float | None
    count_a: int
    count_b: int
    average_a: float
    average_b: float


class CompareResult(BaseModel):
    period_a: str  # "2025-09-01..2025-09-30": *_a fields below belong to this period
    period_b: str
    currency: Currency
    total_a: float
    total_b: float
    delta: float
    delta_pct: float | None
    groups: list[GroupDelta]


class Subscription(BaseModel):
    merchant: str
    category: str
    last_amount_uah: float
    operation_amount: float
    operation_currency: str
    months_charged: int
    first_charge: str
    last_charge: str
    signals: list[str]


class SubscriptionsResult(BaseModel):
    monthly_total_uah: float
    subscriptions: list[Subscription]
    irregular_digital_charges: list[Transaction]


class Rate(BaseModel):
    pair: str
    buy: float | None
    sell: float | None
    cross: float | None
    date: str


class RatesResult(BaseModel):
    rates: list[Rate]


class ForecastResult(BaseModel):
    category: str
    baseline_period: str
    baseline_monthly_spend: float
    monthly_saving: float
    total_saving: float
    months: int
    cut_percent: float
