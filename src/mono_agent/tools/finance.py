"""Tool implementations over the SQLite cache.

SQL is built from fixed fragments plus bound parameters only: the LLM picks validated
arguments, never SQL text. Amounts are kopiykas in SQL and UAH/USD/EUR (2 decimals) outside.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import pairwise
from statistics import median
from zoneinfo import ZoneInfo

from mono_agent.guardrails import sanitize
from mono_agent.mono.cache import Cache
from mono_agent.mono.client import MonoClient
from mono_agent.tools.categories import NON_SPENDING, category
from mono_agent.tools.schemas import (
    AggregateInput,
    AggregateResult,
    CompareInput,
    CompareResult,
    ForecastInput,
    ForecastResult,
    Group,
    GroupDelta,
    NoInput,
    Period,
    QueryInput,
    Rate,
    RatesResult,
    Subscription,
    SubscriptionsInput,
    SubscriptionsResult,
    SyncInput,
    SyncResult,
    Transaction,
    TransactionList,
)

KYIV = ZoneInfo("Europe/Kyiv")
CODES = {980: "UAH", 840: "USD", 978: "EUR", 985: "PLN"}
_SPENDING = f"category(mcc, description) NOT IN ({','.join('?' * len(NON_SPENDING))})"

# Known overlapping services: two active subscriptions in one group are worth a look.
SERVICE_GROUPS = {
    "music streaming": ("spotify", "youtube premium", "apple music", "deezer"),
    "video streaming": ("netflix", "megogo", "sweet.tv", "youtube premium", "oll.tv"),
}


def _bounds(p: Period) -> tuple[int, int]:
    """Inclusive local dates -> unix seconds [start of date_from, end of date_to] in Kyiv."""
    start = datetime(p.date_from.year, p.date_from.month, p.date_from.day, tzinfo=KYIV)
    end = datetime(p.date_to.year, p.date_to.month, p.date_to.day, tzinfo=KYIV) + timedelta(1)
    return int(start.timestamp()), int(end.timestamp()) - 1


def _period_key(t: int, granularity: str) -> str:
    d = datetime.fromtimestamp(t, KYIV)
    if granularity == "month":
        return d.strftime("%Y-%m")
    if granularity == "week":
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"
    return d.strftime("%Y-%m-%d")


def _money(kop: int | float) -> float:
    return round(kop / 100, 2)


def _pct(a: float, b: float) -> float | None:
    return round((b - a) / a * 100, 1) if a else None


@dataclass
class ToolContext:
    client: MonoClient
    cache: Cache
    today: date  # "now" for the agent; fixed in synthetic mode so answers are reproducible
    _accounts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache.db.create_function("category", 2, category, deterministic=True)
        self.cache.db.create_function("period_key", 2, _period_key, deterministic=True)

    def account(self, currency: str) -> str:
        if not self._accounts:
            for a in self.cache.client_info(self.client).accounts:
                # First account per currency wins: the main card for UAH.
                self._accounts.setdefault(CODES.get(a.currency_code, str(a.currency_code)), a.id)
        if currency not in self._accounts:
            raise ValueError(f"no {currency} account")
        return self._accounts[currency]

    def synced(self, p: Period, currency: str) -> tuple[str, int, int]:
        account = self.account(currency)
        start, end = _bounds(p)
        self.cache.sync_statement(self.client, account, start, end, now=time.time())
        # A whole period without a single transaction almost always means a wrong period
        # (e.g. the model picked the wrong year). Say so instead of returning a silent zero.
        db = self.cache.db
        if not db.execute(
            "SELECT 1 FROM transactions WHERE account_id = ? AND time BETWEEN ? AND ? LIMIT 1",
            (account, start, end),
        ).fetchone():
            first, last = db.execute(
                "SELECT MIN(time), MAX(time) FROM transactions WHERE account_id = ?", (account,)
            ).fetchone()
            available = (
                f"{_period_key(first, 'day')}..{_period_key(last, 'day')}" if first else "none"
            )
            raise ValueError(
                f"no {currency} transactions at all between {p.date_from} and {p.date_to}; "
                f"cached data covers {available}. Today is {self.today}: check the year."
            )
        return account, start, end


# --- tools -------------------------------------------------------------------------------


def sync_statement(ctx: ToolContext, args: SyncInput) -> SyncResult:
    account = ctx.account(args.currency)
    start, end = _bounds(args)
    fetched = ctx.cache.sync_statement(ctx.client, account, start, end, now=time.time())
    (count,) = ctx.cache.db.execute(
        "SELECT COUNT(*) FROM transactions WHERE account_id = ? AND time BETWEEN ? AND ?",
        (account, start, end),
    ).fetchone()
    return SyncResult(fetched_from_api=fetched, cached_transactions=count)


def _where(args: QueryInput | AggregateInput, account: str, start: int, end: int):
    clauses, params = ["account_id = ?", "time BETWEEN ? AND ?"], [account, start, end]
    if args.category:
        clauses.append("category(mcc, description) = ?")
        params.append(args.category)
    if isinstance(args, AggregateInput):
        clauses.append(_SPENDING)
        params.extend(NON_SPENDING)
        return clauses, params
    if args.merchant:
        clauses.append("description LIKE ? ESCAPE '\\'")
        escaped = args.merchant.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f"%{escaped}%")
    if args.kind == "spending":
        clauses.append(_SPENDING)
        params.extend(NON_SPENDING)
    elif args.kind == "income":
        clauses.append("amount > 0 AND category(mcc, description) IN ('transfers', 'cashback')")
    if args.min_amount is not None:
        clauses.append("ABS(amount) >= ?")
        params.append(round(args.min_amount * 100))
    if args.max_amount is not None:
        clauses.append("ABS(amount) <= ?")
        params.append(round(args.max_amount * 100))
    return clauses, params


def _transaction(row, currency: str) -> Transaction:
    description, s1 = sanitize(row["description"])
    comment, s2 = sanitize(row["comment"])
    return Transaction(
        time=datetime.fromtimestamp(row["time"], KYIV).strftime("%Y-%m-%d %H:%M"),
        description=description,
        category=category(row["mcc"], row["description"]),
        amount=_money(row["amount"]),
        currency=currency,  # type: ignore[arg-type]
        operation_amount=_money(row["operation_amount"]),
        operation_currency=CODES.get(row["currency_code"], str(row["currency_code"])),
        cashback=_money(row["cashback_amount"]),
        hold=bool(row["hold"]),
        comment=comment,
        suspicious_text_removed=s1 or s2,
    )


def query_transactions(ctx: ToolContext, args: QueryInput) -> TransactionList:
    account, start, end = ctx.synced(args, args.currency)
    clauses, params = _where(args, account, start, end)
    where = " AND ".join(clauses)
    db = ctx.cache.db
    count, total = db.execute(
        f"SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM transactions WHERE {where}", params
    ).fetchone()
    order = "ABS(amount) DESC" if args.order_by == "amount" else "time DESC"
    rows = db.execute(
        f"SELECT * FROM transactions WHERE {where} ORDER BY {order} LIMIT ?",
        [*params, args.limit],
    ).fetchall()
    items = [_transaction(r, args.currency) for r in rows]
    return TransactionList(
        total_count=count, total_amount=_money(total), shown=len(items), items=items
    )


def _group_expr(group_by: str) -> str:
    return {
        "category": "category(mcc, description)",
        "merchant": "description",
    }.get(group_by, f"period_key(time, '{group_by}')")  # group_by is a validated Literal


def aggregate_spending(ctx: ToolContext, args: AggregateInput) -> AggregateResult:
    account, start, end = ctx.synced(args, args.currency)
    clauses, params = _where(args, account, start, end)
    key = _group_expr(args.group_by)
    rows = ctx.cache.db.execute(
        f"SELECT {key} AS k, -SUM(amount) AS total, COUNT(*) AS n, "
        f"SUM(cashback_amount) AS cashback FROM transactions "
        f"WHERE {' AND '.join(clauses)} GROUP BY k",
        params,
    ).fetchall()
    order = (lambda g: g.key) if args.group_by in ("month", "week", "day") else (lambda g: -g.total)
    groups = sorted(
        (
            Group(
                key=sanitize(r["k"])[0] or "",
                total=_money(r["total"]),
                count=r["n"],
                average=_money(r["total"] / r["n"]),
                cashback=_money(r["cashback"]),
            )
            for r in rows
        ),
        key=order,
    )
    return AggregateResult(
        currency=args.currency,
        total_spent=_money(sum(r["total"] for r in rows)),
        total_cashback=_money(sum(r["cashback"] for r in rows)),
        groups=groups[: args.top_n],
    )


def compare_periods(ctx: ToolContext, args: CompareInput) -> CompareResult:
    def run(p: Period) -> dict[str, Group]:
        result = aggregate_spending(
            ctx,
            AggregateInput(
                date_from=p.date_from,
                date_to=p.date_to,
                group_by=args.group_by,
                currency=args.currency,
                category=args.category,
                top_n=50,
            ),
        )
        return {g.key: g for g in result.groups}

    a, b = run(args.period_a), run(args.period_b)
    empty = Group(key="", total=0, count=0, average=0)
    groups = [
        GroupDelta(
            key=k,
            total_a=a.get(k, empty).total,
            total_b=b.get(k, empty).total,
            delta=round(b.get(k, empty).total - a.get(k, empty).total, 2),
            delta_pct=_pct(a.get(k, empty).total, b.get(k, empty).total),
            count_a=a.get(k, empty).count,
            count_b=b.get(k, empty).count,
            average_a=a.get(k, empty).average,
            average_b=b.get(k, empty).average,
        )
        for k in a.keys() | b.keys()
    ]
    groups.sort(key=lambda g: -abs(g.delta))
    total_a = round(sum(g.total for g in a.values()), 2)
    total_b = round(sum(g.total for g in b.values()), 2)
    return CompareResult(
        period_a=f"{args.period_a.date_from}..{args.period_a.date_to}",
        period_b=f"{args.period_b.date_from}..{args.period_b.date_to}",
        currency=args.currency,
        total_a=total_a,
        total_b=total_b,
        delta=round(total_b - total_a, 2),
        delta_pct=_pct(total_a, total_b),
        groups=groups,
    )


def detect_subscriptions(ctx: ToolContext, args: SubscriptionsInput) -> SubscriptionsResult:
    """Monthly charges with a stable price; plus signals that a subscription deserves a look."""
    account, start, end = ctx.synced(args, "UAH")
    rows = ctx.cache.db.execute(
        f"SELECT * FROM transactions WHERE account_id = ? AND time BETWEEN ? AND ? "
        f"AND amount < 0 AND {_SPENDING} ORDER BY time",
        [account, start, end, *NON_SPENDING],
    ).fetchall()
    by_merchant = defaultdict(list)
    for r in rows:
        by_merchant[r["description"]].append(r)

    subs: list[Subscription] = []
    irregular: list[Transaction] = []
    for merchant, charges in by_merchant.items():
        months = {_period_key(c["time"], "month") for c in charges}
        gaps = [(b["time"] - a["time"]) / 86_400 for a, b in pairwise(charges)]
        amounts = [-c["operation_amount"] for c in charges]  # positive: these are debits
        monthly = (
            len(months) >= 3
            and len(charges) <= len(months) * 1.2
            and gaps
            and 25 <= median(gaps) <= 35
            and max(amounts) <= min(amounts) * 1.5  # stable price; allows a price change
        )
        if not monthly:
            if category(charges[0]["mcc"], merchant) == "digital" and len(months) <= 2:
                irregular.extend(_transaction(c, "UAH") for c in charges)
            continue
        first, last = charges[0], charges[-1]
        signals = []
        if first["operation_amount"] != last["operation_amount"]:
            signals.append(
                f"price changed from {_money(-first['operation_amount'])} to "
                f"{_money(-last['operation_amount'])} {CODES.get(last['currency_code'], '')}"
            )
        if (end - last["time"]) / 86_400 > 40:
            signals.append(f"no charge since {_period_key(last['time'], 'day')} (cancelled?)")
        subs.append(
            Subscription(
                merchant=sanitize(merchant)[0] or "",
                category=category(last["mcc"], merchant),
                last_amount_uah=_money(-last["amount"]),
                operation_amount=_money(-last["operation_amount"]),
                operation_currency=CODES.get(last["currency_code"], str(last["currency_code"])),
                months_charged=len(months),
                first_charge=_period_key(first["time"], "day"),
                last_charge=_period_key(last["time"], "day"),
                signals=signals,
            )
        )

    active = [s for s in subs if not any(x.startswith("no charge") for x in s.signals)]
    for group, names in SERVICE_GROUPS.items():
        members = [s for s in active if any(n in s.merchant.lower() for n in names)]
        for s in members:
            others = [m.merchant for m in members if m is not s]
            if others:
                s.signals.append(f"overlaps with {', '.join(others)} ({group})")
    subs.sort(key=lambda s: -s.last_amount_uah)
    return SubscriptionsResult(
        monthly_total_uah=round(sum(s.last_amount_uah for s in active), 2),
        subscriptions=subs,
        irregular_digital_charges=irregular,
    )


def currency_rates(ctx: ToolContext, args: NoInput) -> RatesResult:
    rates = ctx.cache.currency_rates(ctx.client)
    wanted = {840, 978, 985}
    return RatesResult(
        rates=[
            Rate(
                pair=f"{CODES[r.currency_code_a]}/{CODES[r.currency_code_b]}",
                buy=r.rate_buy,
                sell=r.rate_sell,
                cross=r.rate_cross,
                date=_period_key(r.date, "day"),
            )
            for r in rates
            if r.currency_code_a in wanted and r.currency_code_b in CODES
        ]
    )


def forecast_savings(ctx: ToolContext, args: ForecastInput) -> ForecastResult:
    """Baseline = average of the last N full calendar months before `today`."""
    first_of_next = (ctx.today + timedelta(days=1)).replace(day=1)
    end = first_of_next - timedelta(days=1)  # last day of the last full month
    start = end.replace(day=1)
    for _ in range(args.baseline_months - 1):
        start = (start - timedelta(days=1)).replace(day=1)
    spent = aggregate_spending(
        ctx,
        AggregateInput(date_from=start, date_to=end, group_by="category", category=args.category),
    ).total_spent
    monthly = round(spent / args.baseline_months, 2)
    saving = round(monthly * args.cut_percent / 100, 2)
    return ForecastResult(
        category=args.category,
        baseline_period=f"{start:%Y-%m} to {end:%Y-%m}",
        baseline_monthly_spend=monthly,
        monthly_saving=saving,
        total_saving=round(saving * args.months, 2),
        months=args.months,
        cut_percent=args.cut_percent,
    )
