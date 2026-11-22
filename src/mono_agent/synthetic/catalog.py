"""Merchants and fixed payments used by the generator. Names are real Ukrainian brands so the
data reads naturally; people and companies (employer, landlord, friends) are fictional."""

UAH, USD, EUR, PLN = 980, 840, 978, 985

# category -> (merchants as (name, mcc), min UAH, max UAH)
EVERYDAY: dict[str, tuple[list[tuple[str, int]], float, float]] = {
    "groceries": (
        [("Сільпо", 5411), ("АТБ-Маркет", 5411), ("Novus", 5411), ("Varus", 5411), ("Фора", 5411)],
        120,
        900,
    ),
    "cafes": (
        [("Aroma Kava", 5814), ("Львівські круасани", 5814), ("McDonald's", 5814), ("KFC", 5814)],
        60,
        260,
    ),
    "restaurants": (
        [("Пузата Хата", 5812), ("Мафія", 5812), ("Сушия", 5812), ("Хінкальня", 5812)],
        400,
        1400,
    ),
    "taxi": ([("Uklon", 4121), ("Bolt", 4121)], 110, 380),
    "fuel": ([("WOG", 5541), ("OKKO", 5541), ("SOCAR", 5541)], 1200, 1900),
    "pharmacy": (
        [("Аптека АНЦ", 5912), ("Подорожник", 5912), ("Аптека Доброго Дня", 5912)],
        150,
        900,
    ),
    "transport": ([("Київ Цифровий", 4111)], 8, 8),
    "delivery": ([("Нова Пошта", 4215)], 70, 180),
    "electronics": ([("Rozetka", 5732), ("Comfy", 5732), ("Фокстрот", 5732)], 400, 4500),
    "home": ([("Епіцентр", 5200)], 300, 3500),
    "clothing": ([("Intertop", 5661), ("LC Waikiki", 5651)], 700, 4200),
    "cinema": ([("Multiplex", 7832)], 300, 520),
    "pets": ([("Masterzoo", 5995)], 300, 1500),
}

# Base daily probability of at least one purchase in the category.
DAILY_P = {
    "groceries": 0.55,
    "cafes": 0.30,
    "restaurants": 0.14,
    "pharmacy": 0.06,
    "transport": 0.15,
    "delivery": 0.08,
    "electronics": 0.02,
    "home": 0.02,
    "clothing": 0.02,
    "cinema": 0.03,
    "pets": 0.04,
}

# Expected taxi rides per month (more in cold/rainy months, fewer in summer).
TAXI_PER_MONTH = {1: 9, 2: 8, 3: 6, 4: 5, 5: 4, 6: 4, 7: 4, 8: 3, 9: 6, 10: 9, 11: 9, 12: 11}

# Cashback rates by MCC (the "chosen categories" of a monobank black card).
CASHBACK = {4121: 0.05, 5812: 0.05, 5814: 0.05, 5912: 0.02}

# Monthly subscriptions: name, mcc, amount, currency, day of month, active months.
SUBSCRIPTIONS: list[tuple[str, int, float, int, int, range]] = [
    ("Spotify", 5815, 189.00, UAH, 3, range(1, 13)),
    ("Kyivstar", 4814, 250.00, UAH, 7, range(1, 13)),
    ("MEGOGO", 4899, 199.00, UAH, 9, range(1, 7)),  # cancelled after June
    ("Воля", 4899, 220.00, UAH, 12, range(1, 13)),
    ("Netflix.com", 4899, 9.99, USD, 15, range(1, 8)),
    ("Netflix.com", 4899, 11.99, USD, 15, range(8, 13)),  # price increase in August
    ("Apple.com/bill", 5818, 0.99, USD, 18, range(1, 13)),
    ("Google YouTube Premium", 5815, 149.00, UAH, 20, range(1, 13)),  # overlaps Spotify
    ("OpenAI *ChatGPT Subscr", 5734, 20.00, USD, 25, range(3, 13)),
    ("Sport Life", 7997, 1350.00, UAH, 1, range(1, 13)),
]

EMPLOYER = ("ТОВ «Альфа Технології»", "43215678")
LANDLORD = "Олександр П."
FRIENDS = ["Андрій М.", "Олена К.", "Марія С.", "Дмитро Л.", "Ірина В."]
