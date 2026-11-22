"""Mask card numbers and IBANs before text reaches an LLM, a trace or the cache."""

import re

# 13-19 digits, optionally grouped by spaces or dashes (how PANs appear in free text).
_PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# Ukrainian IBAN: UA + 2 check digits + 25 alphanumerics; allow spaces between groups.
_IBAN = re.compile(r"\bUA\d{2}(?:\s?[0-9A-Z]){25}\b", re.IGNORECASE)


def mask_pan(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return f"****{digits[-4:]}"


def mask_iban(match: re.Match[str]) -> str:
    compact = re.sub(r"\s", "", match.group())
    return f"{compact[:2].upper()}****{compact[-4:]}"


def mask_text(text: str | None) -> str | None:
    """Replace every PAN/IBAN in text with a masked form keeping the last 4 characters."""
    if not text:
        return text
    return _PAN.sub(mask_pan, _IBAN.sub(mask_iban, text))
