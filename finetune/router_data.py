"""Training data for the fine-tuned router: template questions in Ukrainian and English.

    uv run python -m finetune.router_data      # -> finetune/data/{train,valid}.jsonl

The eval questions are the test set, so they never appear here: templates were written from
the intent definitions, not from the eval, and any generated question whose wording is close
to an eval question (word Jaccard >= 0.5) is dropped. Topics still overlap (both ask about
taxi spending in October): that is the task, not a leak.
"""

import json
import random
import re
from itertools import product
from pathlib import Path

INTENTS = ["transactions", "knowledge", "mixed", "export", "out_of_scope"]
# The plan is a function of the intent (the same mapping the router prompt spells out), so the
# small model only has to pick the intent.
STEPS = {
    "transactions": ["transactions"],
    "knowledge": ["knowledge"],
    "mixed": ["transactions", "knowledge"],
    "export": ["transactions", "export"],
    "out_of_scope": [],
}
SYSTEM = (
    "Classify the message for a monobank personal-finance assistant. "
    "Reply with one label: transactions, knowledge, mixed, export or out_of_scope."
)
MAX_SIMILARITY = 0.5
PER_INTENT = 260
OUT = Path(__file__).parent / "data"

# --- slots: (Ukrainian accusative after "на", English object, English adjective) ------------
CATS = [
    ("таксі", "taxis", "taxi"),
    ("продукти", "groceries", "grocery"),
    ("кафе й ресторани", "eating out", "restaurant"),
    ("пальне", "fuel", "fuel"),
    ("ліки", "medicine", "pharmacy"),
    ("комунальні послуги", "utilities", "utility"),
    ("одяг", "clothes", "clothing"),
    ("розваги", "entertainment", "entertainment"),
    ("доставку їжі", "food delivery", "food delivery"),
    ("спортзал", "the gym", "gym"),
    ("мобільний зв'язок", "my phone plan", "phone"),
    ("подорожі", "travel", "travel"),
    ("каву", "coffee", "coffee"),
    ("зоотовари", "pet supplies", "pet"),
]
PERIODS_UK = [
    "у січні", "в лютому", "у березні", "у квітні", "у травні", "в червні", "в липні",
    "у серпні", "у вересні", "в жовтні", "у листопаді", "у грудні", "минулого місяця",
    "за останні два тижні", "цього кварталу", "за перше півріччя", "за весь рік",
    "минулого тижня", "взимку", "влітку",
]  # fmt: skip
PERIODS_EN = [
    "in January", "in February", "in March", "in April", "in May", "in June", "in July",
    "in August", "in September", "in October", "in November", "in December", "last month",
    "over the past two weeks", "this quarter", "in the first half of the year", "this year",
    "last week", "over the winter", "over the summer",
]  # fmt: skip
MERCHANTS = [
    "Сільпо", "АТБ", "Novus", "Bolt", "Uklon", "Glovo", "Rozetka", "Comfy", "Епіцентр",
    "WOG", "OKKO", "Aroma Kava", "McDonald's", "Нова пошта",
]  # fmt: skip
SUBS = [
    "Netflix", "Spotify", "YouTube Premium", "iCloud", "Google One", "Duolingo", "Megogo",
    "Sweet.tv", "ChatGPT Plus", "Adobe",
]  # fmt: skip
AMOUNTS = ["120", "349", "1 200", "75,50", "2 499", "560"]
DATES_UK = ["3 березня", "17 червня", "вчора", "минулої п'ятниці", "12 вересня"]
DATES_EN = ["on March 3", "on June 17", "yesterday", "last Friday", "on September 12"]
PCTS = ["10", "20", "25", "40", "50"]
MONTHS_N = ["2", "4", "6", "12"]

TEMPLATES: dict[str, list[str]] = {
    "transactions": [
        # Ukrainian
        "Скільки я витратив на {cat_uk} {per_uk}?",
        "Скільки грошей пішло на {cat_uk} {per_uk}?",
        "Покажи мої витрати на {cat_uk} {per_uk}",
        "Чому {per_uk} я витратив на {cat_uk} більше, ніж зазвичай?",
        "Порівняй мої витрати на {cat_uk} {per_uk} і {per2_uk}",
        "Скільки разів я платив у {merchant} {per_uk}?",
        "Яка в мене середня сума чека в {merchant}?",
        "Скільки я загалом витратив {per_uk}?",
        "На що я витрачаю найбільше {per_uk}?",
        "Де я найбільше витрачав {per_uk}? Дай топ-5 магазинів",
        "Чи є в мене підписки, про які я забув?",
        "Скільки я плачу за {sub} щомісяця?",
        "Коли востаннє списували гроші за {sub}?",
        "Чи подорожчала підписка {sub}?",
        "Скільки кешбеку мені нарахували {per_uk}?",
        "Які повернення коштів я отримав {per_uk}?",
        "Скільки я відклав у банки {per_uk}?",
        "Скільки мені прийшло зарплати {per_uk}?",
        "Від кого я отримував перекази {per_uk}?",
        "Скільки доларів я купив {per_uk}?",
        "Почому сьогодні долар у монобанку?",
        "Який зараз курс євро?",
        "Якщо я витрачатиму на {cat_uk} на {pct}% менше, скільки зекономлю за {n} міс.?",
        "Скільки я відкладу, якщо скорочу витрати на {cat_uk} на {pct}%?",
        "Що за списання на {amount} грн {date_uk}?",
        "Яка моя найдорожча покупка {per_uk}?",
        "Скільки я витратив у {merchant} {per_uk}?",
        "Чому {per_uk} мої витрати виросли?",
        "Скільки я заплатив за оренду {per_uk}?",
        "Чи повернули мені гроші за замовлення в {merchant}?",
        "Скільки я в середньому витрачаю на {cat_uk} щомісяця?",
        # English
        "How much have I spent on {cat_en} {per_en}?",
        "Show my {cat_adj} expenses {per_en}",
        "Why did my {cat_adj} spending go up {per_en}?",
        "Compare what I spent on {cat_en} {per_en} versus {per2_en}",
        "How many times did I pay at {merchant} {per_en}?",
        "What's my average purchase at {merchant}?",
        "What were my total expenses {per_en}?",
        "Where does most of my money go?",
        "List the merchants I spent the most at {per_en}",
        "Do I have any subscriptions I forgot about?",
        "How much is {sub} costing me each month?",
        "When was I last charged for {sub}?",
        "Did the price of {sub} go up?",
        "How much cashback did I earn {per_en}?",
        "Which refunds did I get {per_en}?",
        "How much did I save into jars {per_en}?",
        "How much salary came in {per_en}?",
        "Who sent me money {per_en}?",
        "How many euros did I buy {per_en}?",
        "What's the euro rate right now?",
        "What's monobank's dollar rate today?",
        "If I spend {pct}% less on {cat_en}, how much will I save in {n} months?",
        "What is the {amount} UAH charge from {date_en}?",
        "What was my most expensive purchase {per_en}?",
        "How much went to {merchant} {per_en}?",
        "Why were my expenses higher {per_en}?",
        "How much rent did I pay {per_en}?",
        "Did {merchant} refund my order?",
        "On average, how much do I spend on {cat_en} per month?",
    ],
    "knowledge": [
        "Скільки коштує обслуговування картки монобанку?",
        "Як збільшити кредитний ліміт?",
        "Що таке «Покупка частинами» і як вона працює?",
        "Як перевипустити картку, якщо закінчується термін дії?",
        "Чи можна відкрити картку дитині?",
        "Які ліміти на перекази з картки на картку?",
        "Як працює кешбек на АЗС?",
        "Як закрити банку і куди підуть гроші?",
        "Чи є комісія за поповнення картки готівкою?",
        "Яка комісія за оплату карткою за кордоном?",
        "Як отримати віртуальну картку?",
        "Як змінити категорії кешбеку на наступний місяць?",
        "Чи можна платити карткою монобанку через Apple Pay?",
        "За яким курсом конвертується оплата в доларах?",
        "Як працює овердрафт?",
        "Чи можна мати кілька банок одночасно?",
        "Як підвищити ліміт на зняття готівки?",
        "Які умови депозиту в монобанку?",
        "Чи стягується плата за SMS-сповіщення?",
        "Чим відрізняється чорна картка від білої?",
        "Як змінити PIN-код картки?",
        "Чи можна достроково погасити розстрочку?",
        "Як відкрити рахунок ФОП?",
        "Чи можна відкрити валютну картку в євро?",
        "Як отримати довідку про стан рахунку?",
        "What does monobank charge for card maintenance?",
        "How can I raise my credit limit?",
        "How does pay-in-installments work?",
        "How do I reissue a card that is about to expire?",
        "Can I open a card for my child?",
        "What are the limits on card-to-card transfers?",
        "How is cashback calculated at gas stations?",
        "How do I close a jar, and where does the money go?",
        "Is there a fee for topping up the card with cash?",
        "What's the fee for paying by card abroad?",
        "How do I get a virtual card?",
        "How do I change my cashback categories for next month?",
        "Can I use my monobank card with Google Pay?",
        "Which exchange rate is used when I pay in dollars?",
        "How does the overdraft work?",
        "Can I have several jars at once?",
        "How do I raise my ATM withdrawal limit?",
        "What are the terms of monobank deposits?",
        "Do SMS notifications cost anything?",
        "What's the difference between the black and the white card?",
        "How do I change my card PIN?",
        "Can I repay an installment plan early?",
        "How do I open a business account as a sole proprietor?",
        "Can I get a euro card?",
        "How do I get an account statement certificate?",
    ],
    "mixed": [
        "Скільки я витратив за кордоном {per_uk} і яка комісія за оплату в іноземній валюті?",
        "Скільки я витратив на {cat_uk} {per_uk} і чи є на це підвищений кешбек?",
        "Скільки разів я поповнював картку готівкою {per_uk} і чи це безкоштовно?",
        "Скільки я заплатив за покупки частинами {per_uk} і чи можна погасити їх достроково?",
        "Я переказав {amount} грн на картку іншого банку {date_uk}, чи мала бути комісія?",
        "Скільки коштували мої підписки {per_uk} і чи нараховується на них кешбек?",
        "Скільки я витратив з кредитного ліміту {per_uk} і як нараховуються відсотки?",
        "Скільки я заплатив за обслуговування картки {per_uk} і чи це відповідає тарифу?",
        "How much did I spend abroad {per_en}, and what fee applies to foreign payments?",
        "What did I spend on {cat_en} {per_en}, and is there a higher cashback for it?",
        "How many cash top-ups did I make {per_en}, and are they free?",
        "How much did I pay in installments {per_en}, and can I repay them early?",
        "I sent {amount} UAH to another bank's card {date_en}; should there be a fee?",
        "How much did my subscriptions cost {per_en}, and do they earn cashback?",
        "How much did I spend from my credit limit {per_en}, and how is interest charged?",
        "Was I charged a card fee {per_en}, and is that in line with the tariff?",
    ],
    "export": [
        "Експортуй мої витрати {per_uk} у файл",
        "Збережи виписку {per_uk} як звіт",
        "Сформуй звіт по категоріях {per_uk} і збережи його",
        "Вивантаж список транзакцій у {merchant} {per_uk}",
        "Збережи у файл звіт про кешбек {per_uk}",
        "Зроби файл зі звітом про мої доходи {per_uk}",
        "Підготуй файл зі звітом про витрати на {cat_uk} {per_uk}",
        "Хочу CSV з моїми покупками {per_uk}",
        "Export my transactions {per_en} to a file",
        "Save a report of my {cat_adj} spending {per_en}",
        "Download my statement {per_en} as a report",
        "Make a spending report by merchant {per_en} and save it",
        "Create a file with my income {per_en}",
        "I need a CSV of my purchases {per_en}",
        "Save my {merchant} transactions {per_en} to a file",
    ],
    "out_of_scope": [
        "Розкажи анекдот",
        "Напиши вірш про осінь",
        "Який фільм подивитися ввечері?",
        "Як приготувати борщ?",
        "Переклади англійською: я люблю каву",
        "Хто написав «Кобзар»?",
        "Скільки буде 17 помножити на 23?",
        "Порадь, яку криптовалюту купити зараз",
        "Чи варто інвестувати в золото?",
        "Яка погода буде у Львові на вихідних?",
        "Як вилікувати застуду?",
        "Напиши код на Python для сортування списку",
        "Хто переміг на Євробаченні?",
        "Порекомендуй книжку про космос",
        "Як вивчити англійську за місяць?",
        "Склади план тренувань у залі",
        "Який телефон краще купити?",
        "Розкажи про історію Києва",
        "Яка столиця Австралії?",
        "Придумай назву для кав'ярні",
        "В які акції вкласти 10 тисяч доларів?",
        "Що думаєш про біткоїн?",
        "Напиши привітання з днем народження для мами",
        "Як доїхати до Карпат?",
        "Чому небо блакитне?",
        "Склади мені резюме",
        "Котра година в Нью-Йорку?",
        "Порадь серіал на вечір",
        "Як почистити кавоварку від накипу?",
        "Уяви, що ти ворожка, і передбач моє майбутнє",
        "Tell me a joke",
        "Write a haiku about rain",
        "What's a good movie for tonight?",
        "How do I make pancakes?",
        "Translate 'thank you very much' into Spanish",
        "Who painted the Mona Lisa?",
        "What is 144 divided by 12?",
        "Which crypto should I buy right now?",
        "Is it a good time to invest in gold?",
        "What's the weather in Odesa this weekend?",
        "How do I get rid of a cold?",
        "Write a Python function that reverses a string",
        "Recommend a book about space",
        "How can I learn English fast?",
        "Make me a gym workout plan",
        "Which phone should I buy?",
        "Tell me about the history of Lviv",
        "What's the capital of Canada?",
        "Suggest a name for a coffee shop",
        "Which stocks should I put $10,000 into?",
        "What do you think about Bitcoin?",
        "Write a birthday message for my dad",
        "How do I get to the Carpathians?",
        "Why is the sky blue?",
        "Write my CV",
        "What time is it in Tokyo?",
        "Recommend a TV series",
        "How do I descale a coffee machine?",
        "Pretend you are a fortune teller and predict my future",
        "Explain quantum computing in simple terms",
    ],
}


def _fills(template: str) -> list[str]:
    """Every combination of the slots this template uses."""
    slots = sorted(set(re.findall(r"{(\w+)}", template)))
    values = {
        "cat_uk": [c[0] for c in CATS],
        "cat_en": [c[1] for c in CATS],
        "cat_adj": [c[2] for c in CATS],
        "per_uk": PERIODS_UK,
        "per2_uk": PERIODS_UK,
        "per_en": PERIODS_EN,
        "per2_en": PERIODS_EN,
        "merchant": MERCHANTS,
        "sub": SUBS,
        "amount": AMOUNTS,
        "date_uk": DATES_UK,
        "date_en": DATES_EN,
        "pct": PCTS,
        "n": MONTHS_N,
    }
    out = []
    for combo in product(*(values[s] for s in slots)):
        fill = dict(zip(slots, combo, strict=True))
        if fill.get("per_uk", 0) == fill.get("per2_uk", 1) or fill.get("per_en", 0) == fill.get(
            "per2_en", 1
        ):
            continue
        out.append(template.format(**fill))
    return out


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def _variant(text: str, rng: random.Random) -> str:
    """People type in lowercase and skip the question mark; teach the model not to care."""
    if rng.random() < 0.3:
        text = text.rstrip("?")
    if rng.random() < 0.2:
        text = text.lower()
    return text


def build(test_questions: list[str], seed: int = 0) -> list[tuple[str, str]]:
    """(question, intent) pairs, the same number of fills from every template of an intent,
    so a template with many slot combinations doesn't crowd out the others."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent in INTENTS:
        templates = TEMPLATES[intent]
        k = -(-PER_INTENT // len(templates))  # ceil
        chosen: set[str] = set()
        for t in templates:
            fills = _fills(t)
            chosen |= set(rng.sample(fills, min(k, len(fills))))
        if len(chosen) < PER_INTENT:  # fixed-text templates: add the lowercase/no-"?" forms
            chosen |= {q.lower() for q in chosen} | {q.rstrip("?") for q in chosen}
        for q in rng.sample(sorted(chosen), min(PER_INTENT, len(chosen))):
            q = _variant(q, rng)
            if max(similarity(q, t) for t in test_questions) < MAX_SIMILARITY:
                rows.append((q, intent))
    rng.shuffle(rows)
    return rows


def to_chat(question: str, intent: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": intent},
        ]
    }


def main() -> None:
    from eval.dataset import load

    tests = [q.q for q in load()]
    rows = build(tests)
    n_valid = len(rows) // 10
    OUT.mkdir(exist_ok=True)
    for name, part in [("valid", rows[:n_valid]), ("train", rows[n_valid:])]:
        with (OUT / f"{name}.jsonl").open("w") as f:
            for q, intent in part:
                f.write(json.dumps(to_chat(q, intent), ensure_ascii=False) + "\n")
    counts = {i: sum(1 for _, x in rows if x == i) for i in INTENTS}
    print(f"{len(rows)} examples ({len(rows) - n_valid} train, {n_valid} valid): {counts}")


if __name__ == "__main__":
    main()
