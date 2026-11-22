"""MCC -> spending category. Registered as a SQLite function so SQL can filter/group by it."""

from typing import Literal, get_args

# Single source of truth for category names: tool schemas use this Literal as an enum.
Category = Literal[
    "groceries", "restaurants", "taxi", "fuel", "car", "health", "transport", "travel",
    "delivery", "electronics", "home", "clothing", "entertainment", "digital", "telecom",
    "utilities", "sport", "beauty", "pets", "cash", "shopping", "other",
    "transfers", "jars", "currency_exchange", "cashback",
]  # fmt: skip
CATEGORIES: tuple[str, ...] = get_args(Category)

_MCC: dict[Category, tuple[int, ...]] = {
    "groceries": (5411, 5422, 5441, 5451, 5462, 5499),
    "restaurants": (5812, 5813, 5814),
    "taxi": (4121,),
    "fuel": (5541, 5542, 5983),
    "car": (5533, 7538, 7542),
    "health": (5912, 8011, 8021, 8062, 8099),
    "transport": (4111, 4131),
    "travel": (3000, 4112, 4411, 4511, 4722, 7011),
    "delivery": (4215,),
    "electronics": (5722, 5732),
    "home": (5200, 5211, 5251, 5712),
    "clothing": (5611, 5621, 5631, 5641, 5651, 5661, 5691, 5699),
    "entertainment": (7832, 7922, 7996),
    "digital": (4899, 5734, 5815, 5816, 5817, 5818),
    "telecom": (4814,),
    "utilities": (4900,),
    "sport": (7941, 7997),
    "beauty": (7230, 7298),
    "pets": (5995,),
    "cash": (6010, 6011),
    "shopping": (5310, 5311, 5331, 5999),
}
MCC_TO_CATEGORY: dict[int, Category] = {mcc: cat for cat, codes in _MCC.items() for mcc in codes}

# Money moving between your own pockets or people, not spending at merchants.
NON_SPENDING = ("transfers", "jars", "currency_exchange", "cashback")
assert set(CATEGORIES) == {*_MCC, "other", *NON_SPENDING}, "category list out of sync"


def category(mcc: int, description: str) -> Category:
    if mcc == 4829:  # "money transfer": split by what monobank writes in the description
        if description.startswith("На банку"):
            return "jars"
        if description.startswith(("Купівля", "Обмін валюти")):
            return "currency_exchange"
        if description.startswith("Виведення кешбеку"):
            return "cashback"
        return "transfers"
    return MCC_TO_CATEGORY.get(mcc, "other")
