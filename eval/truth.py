"""Ground truth computed directly from the synthetic generator's output.

Deliberately independent of the agent's tools: plain Python over StatementItem objects, so a
bug in the tools' SQL cannot also hide in the expected answers. The only shared piece is the
MCC -> category definition, which is a definition, not a computation.
"""

from datetime import date, datetime

from mono_agent.mono.models import StatementItem
from mono_agent.synthetic.generator import KYIV, Dataset, generate
from mono_agent.tools.categories import NON_SPENDING, category

DATA: Dataset = generate(42)
ACCOUNTS = {
    {980: "UAH", 840: "USD", 978: "EUR"}[a.currency_code]: a.id for a in DATA.client_info.accounts
}


def _items(currency: str = "UAH") -> list[StatementItem]:
    return DATA.statements[ACCOUNTS[currency]]


def _in(item: StatementItem, start: str, end: str) -> bool:
    d = datetime.fromtimestamp(item.time, KYIV).date()
    return date.fromisoformat(start) <= d <= date.fromisoformat(end)


def _cat(item: StatementItem) -> str:
    return category(item.mcc, item.description)


def _uah(kop: int) -> float:
    return round(kop / 100, 2)


def _spending(start: str, end: str, cat: str | None = None, currency: str = "UAH"):
    return [
        i
        for i in _items(currency)
        if _in(i, start, end) and _cat(i) not in NON_SPENDING and (cat is None or _cat(i) == cat)
    ]


def spent(start: str, end: str, category: str | None = None, currency: str = "UAH") -> float:
    """Spending net of refunds, transfers/jars/FX/cashback excluded."""
    return _uah(-sum(i.amount for i in _spending(start, end, category, currency)))


def count(start: str, end: str, category: str) -> float:
    return float(sum(1 for i in _spending(start, end, category) if i.amount < 0))


def cashback(start: str, end: str, category: str | None = None) -> float:
    return _uah(sum(i.cashback_amount for i in _spending(start, end, category)))


def merchant(start: str, end: str, name: str) -> float:
    """Net amount (negative = paid) at merchants whose description contains name."""
    return _uah(
        sum(
            i.amount
            for i in _items()
            if _in(i, start, end) and name.lower() in i.description.lower()
        )
    )


def income(start: str, end: str, name: str | None = None) -> float:
    return _uah(
        sum(
            i.amount
            for i in _items()
            if _in(i, start, end)
            and i.amount > 0
            and _cat(i) in ("transfers", "cashback")
            and (name is None or name in i.description)
        )
    )


def refunds(start: str, end: str) -> float:
    return _uah(sum(i.amount for i in _spending(start, end) if i.amount > 0))


def largest(start: str, end: str) -> float:
    return _uah(-min(i.amount for i in _spending(start, end)))


def forecast(category: str, cut_percent: float, months: int) -> float:
    """Same rule the product promises: average of the last 3 full months before today."""
    monthly = round(spent("2025-10-01", "2025-12-31", category) / 3, 2)
    return round(round(monthly * cut_percent / 100, 2) * months, 2)


def received(currency: str, start: str, end: str) -> float:
    """Money that arrived on a USD/EUR account (in that currency)."""
    return _uah(sum(i.amount for i in _items(currency) if _in(i, start, end) and i.amount > 0))


def account_spent(currency: str, start: str, end: str) -> float:
    return _uah(-sum(i.amount for i in _items(currency) if _in(i, start, end) and i.amount < 0))


def rate(pair: str, field: str) -> float:
    a, b = pair.split("/")
    codes = {"USD": 840, "EUR": 978, "PLN": 985, "UAH": 980}
    r = next(
        r for r in DATA.rates if r.currency_code_a == codes[a] and r.currency_code_b == codes[b]
    )
    return float(getattr(r, f"rate_{field}"))


def jar(title: str) -> float:
    return _uah(next(j.balance for j in DATA.client_info.jars if j.title == title))


def last_charge(name: str) -> float:
    charges = [i for i in _items() if i.description == name and i.amount < 0]
    return _uah(-max(charges, key=lambda i: i.time).amount)


def subscriptions_total() -> float:
    """Monthly cost of subscriptions still active at the end of the year."""
    active = [
        "Sport Life",
        "OpenAI *ChatGPT Subscr",
        "Netflix.com",
        "Kyivstar",
        "Воля",
        "Spotify",
        "Google YouTube Premium",
        "Apple.com/bill",
    ]
    return round(sum(last_charge(n) for n in active), 2)


def days(start: str, end: str) -> float:
    return float((date.fromisoformat(end) - date.fromisoformat(start)).days + 1)


FUNCTIONS = {
    f.__name__: f
    for f in [
        spent,
        count,
        cashback,
        merchant,
        income,
        refunds,
        largest,
        forecast,
        received,
        account_spent,
        rate,
        jar,
        last_charge,
        subscriptions_total,
        days,
    ]
}


def resolve(spec: dict) -> float:
    """{"fn": "spent", "start": ..., ...} -> number. Literal numbers pass through."""
    args = {k: v for k, v in spec.items() if k != "fn"}
    return FUNCTIONS[spec["fn"]](**args)
