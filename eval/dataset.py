"""Load and validate eval/dataset/questions.yaml."""

import calendar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from eval import truth

PATH = Path(__file__).parent / "dataset" / "questions.yaml"
Intent = Literal["transactions", "knowledge", "mixed", "export", "out_of_scope", "injection"]


class ExpectedCall(BaseModel):
    name: str | list[str]
    args: dict[str, Any] = Field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [self.name] if isinstance(self.name, str) else self.name


class Evidence(BaseModel):
    doc: str
    text: str


class Question(BaseModel):
    id: str
    q: str
    intent: Intent | list[Intent]
    tags: list[str] = Field(default_factory=list)
    smoke: bool = False
    tools: list[ExpectedCall] = Field(default_factory=list)
    numbers: list[float | dict] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    refuse: bool = False
    reference: str = ""

    @property
    def intents(self) -> list[str]:
        return [self.intent] if isinstance(self.intent, str) else list(self.intent)

    def expected_numbers(self) -> list[list[float]]:
        """One entry per expected number, each a list of acceptable values (usually one)."""
        return [
            [_value(a) for a in n["any"]] if isinstance(n, dict) and "any" in n else [_value(n)]
            for n in self.numbers
        ]


def _value(spec: float | dict) -> float:
    return float(spec) if not isinstance(spec, dict) else truth.resolve(expand(spec))


def period(p: str) -> tuple[str, str]:
    """'2025' | '2025-10' | '2025-10..2025-12' -> (first day, last day)."""
    first, _, last = str(p).partition("..")
    last = last or first

    def bounds(s: str) -> tuple[str, str]:
        if len(s) == 4:
            return f"{s}-01-01", f"{s}-12-31"
        y, m = map(int, s.split("-"))
        return f"{s}-01", f"{s}-{calendar.monthrange(y, m)[1]:02d}"

    return bounds(first)[0], bounds(last)[1]


def expand(spec: dict) -> dict:
    """Replace the `period` shorthand with explicit dates (start/end for truth functions)."""
    if "period" not in spec:
        return spec
    start, end = period(spec["period"])
    rest = {k: v for k, v in spec.items() if k != "period"}
    return {**rest, "start": start, "end": end}


def expand_args(args: dict) -> dict:
    """Same shorthand in expected tool args -> date_from/date_to."""
    if "period" not in args:
        return args
    start, end = period(args["period"])
    rest = {k: v for k, v in args.items() if k != "period"}
    return {**rest, "date_from": start, "date_to": end}


def load(subset: str | None = None) -> list[Question]:
    questions = [Question.model_validate(q) for q in yaml.safe_load(PATH.read_text("utf-8"))]
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    if subset == "smoke":
        return [q for q in questions if q.smoke]
    if subset == "core":
        # Every 4th question: the file is grouped by type, so this is a stratified ~25% sample
        # small enough for rate-limited free API tiers, with every question type represented.
        return questions[::4]
    return questions
