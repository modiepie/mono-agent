"""Deterministic guardrails: prompt-injection detection and number grounding.

Both are plain code on purpose: they must hold even when the model misbehaves.
"""

import re
from collections.abc import Iterable
from typing import Any

_INJECTION = re.compile(
    r"|".join(
        [
            r"ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous\s+|prior\s+|above\s+)?"
            r"(instructions|rules|prompts?)",
            r"disregard\s+(all\s+|the\s+|your\s+)?(previous\s+|prior\s+)?(instructions|rules)",
            r"forget\s+(all\s+|your\s+)?(previous\s+)?instructions",
            r"\bsystem\s*(prompt|:)",
            r"you\s+are\s+now\b",
            r"(print|reveal|show)\s+(me\s+)?(the\s+)?(full|whole|complete)\s+(card|pan|iban)",
            r"ігноруй\w*\s+(\w+\s+)?(інструкці|правил)",
            r"забудь\w*\s+(\w+\s+)?(інструкці|правил)",
            r"игнорируй\w*\s+(\w+\s+)?(инструкци|правил)",
            r"системн\w+\s+(промпт|підказк|інструкці)",
        ]
    ),
    re.IGNORECASE,
)

REMOVED = "[removed: instruction-like text]"


def looks_like_injection(text: str | None) -> bool:
    return bool(text and _INJECTION.search(text))


def sanitize(text: str | None) -> tuple[str | None, bool]:
    """Return (safe text, was_suspicious). Data with instructions in it never reaches a prompt."""
    if looks_like_injection(text):
        return REMOVED, True
    return text, False


# --- number grounding ---------------------------------------------------------------------

# 1 360,96 | 1,360.96 | 1360.96 | -250 | 30% ; spaces as thousands separators are common in UK.
_NUMBER = re.compile(
    r"(?<![\w.,])-?(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,360.96
    r"|\d{1,3}(?:[   ]\d{3})+(?:,\d+)?"  # 1 360,96
    r"|\d+(?:[.,]\d+)?"  # 1360.96 / 1360,96
    r")"
)


def parse_numbers(text: str) -> list[float]:
    out = []
    for raw in _NUMBER.findall(text):
        s = re.sub(r"[   ]", "", raw)
        if "," in s and "." in s:  # 1,360.96 -> comma is a thousands separator
            s = s.replace(",", "")
        elif s.count(",") == 1 and len(s.split(",")[1]) != 3:  # 1360,96 -> decimal comma
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        out.append(float(s))
    return out


def numbers_in(obj: Any) -> set[float]:
    """Every number found anywhere in a JSON-like structure (values and numeric strings)."""
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, int | float):
        found.add(float(obj))
    elif isinstance(obj, str):
        found.update(parse_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= numbers_in(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            found |= numbers_in(v)
    return found


def ungrounded_numbers(answer: str, allowed: Iterable[float]) -> list[float]:
    """Numbers in the answer that no tool result (or the question) supports.

    A number is grounded if it equals an allowed value, or a rounded / absolute version of one
    (answers say "1 361 UAH" for 1360.96 and "250" for -250).
    """
    allowed_abs = {abs(a) for a in allowed}
    bad = []
    for n in parse_numbers(answer):
        n = abs(n)
        # shortcut: small integers (counts, day numbers, "3 months") pass unchecked; a count
        # the model got wrong is caught by the eval's exact-answer checks, not here.
        if n.is_integer() and n <= 31:
            continue
        if any(abs(n - a) < 0.006 or round(a) == n or round(a, 1) == n for a in allowed_abs):
            continue
        bad.append(n)
    return bad
