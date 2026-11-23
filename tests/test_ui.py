"""UI smoke test with Streamlit's AppTest (no browser, no API)."""

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = Path(__file__).parent.parent / "src" / "mono_agent" / "service" / "ui.py"


def test_ui_renders_in_ukrainian_and_handles_missing_api(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(str(UI), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Фінансовий помічник monobank"
    assert any("API недоступне" in e.value for e in at.error)
    assert at.button[0].label == "Нова розмова"


def test_every_visible_string_is_ukrainian():
    """UI text must be Ukrainian: no Latin-only words in user-facing string literals."""
    source = UI.read_text("utf-8")
    ui_calls = re.findall(
        r"(?:st\.\w+|\.button|\.status|\.update|\.write|\.info|\.error|\.success|label=)\("
        r'?\s*f?"([^"]+)"',
        source,
    )
    # chat roles ("user", "assistant") are identifiers, not visible text
    allowed = {"monobank", "mono", "API", "Langfuse", "API_URL", "user", "assistant"}
    for text in ui_calls:
        latin_words = set(re.findall(r"\b[A-Za-z]{3,}\b", re.sub(r"\{[^}]*\}|`[^`]*`", "", text)))
        assert latin_words <= allowed, f"non-Ukrainian UI text: {text!r}"
