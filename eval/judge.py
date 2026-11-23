"""LLM-as-judge for what code can't check: is every claim supported, is the answer useful.

Numbers are NOT judged here (they are checked exactly in metrics.py). The judge sees the same
context the composer saw, plus the dataset's reference note, and scores against a rubric.
"""

import json
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

RUBRIC = """You grade answers of a personal-finance assistant for a monobank customer.
You get the question, the data the assistant had (tool results and documents), a reference
note written by the test author, and the answer.

Score each from 1 to 5:
- faithfulness: are all claims in the answer supported by the data? 5 = everything supported;
  3 = one unsupported or wrong claim; 1 = mostly unsupported or contradicts the data.
  Wrong reasoning counts (e.g. saying there were no car costs when the data shows some).
- helpfulness: does it answer what was asked, clearly, in the user's language, with the key
  numbers or facts? 5 = complete and clear; 3 = partly; 1 = does not answer.

Judge only against the data and the reference note, not your own knowledge of monobank.
Be strict and brief."""


class Verdict(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    reason: str = Field(description="One sentence: the main problem, or 'ok'")


def judge(
    model: BaseChatModel, question: str, context: dict, reference: str, answer: str
) -> Verdict:
    data = json.dumps(context, ensure_ascii=False)
    # shortcut: long contexts are cut to keep the judge call cheap; raise the limit if the
    # judge starts flagging claims that live in the cut part.
    data = data[:15_000]
    prompt = (
        f"Question: {question}\n\nData:\n{data}\n\nReference note: {reference or '-'}"
        f"\n\nAnswer:\n{answer}"
    )
    structured = model.with_structured_output(Verdict)
    return cast(Verdict, structured.invoke([SystemMessage(RUBRIC), HumanMessage(prompt)]))
