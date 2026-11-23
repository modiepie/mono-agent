"""The eval's own logic: metrics, dataset, usage tracking, record/replay."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from eval import dataset, metrics
from eval.dataset import Evidence, ExpectedCall, Question
from eval.recording import NotRecorded, Recording
from mono_agent.providers.usage import UsageTracker
from tests.conftest import Scripted


def test_dataset_loads_and_every_number_resolves():
    questions = dataset.load()
    assert len(questions) >= 60
    for q in questions:
        assert all(isinstance(a, float) for alts in q.expected_numbers() for a in alts)
    assert dataset.load("smoke") and all(q.smoke for q in dataset.load("smoke"))


@pytest.mark.parametrize(
    ("p", "bounds"),
    [
        ("2025", ("2025-01-01", "2025-12-31")),
        ("2025-02", ("2025-02-01", "2025-02-28")),
        ("2025-10..2025-12", ("2025-10-01", "2025-12-31")),
    ],
)
def test_period_shorthand(p, bounds):
    assert dataset.period(p) == bounds


def test_tool_scores_need_name_and_args_subset():
    expected = [
        ExpectedCall(
            name=["aggregate_spending", "query_transactions"],
            args={"period": "2025-10", "category": "taxi"},
        )
    ]
    good = {
        "tool": "query_transactions",
        "ok": True,
        "args": {
            "date_from": "2025-10-01",
            "date_to": "2025-10-31",
            "category": "taxi",
            "limit": 20,
        },
    }
    wrong_year = {**good, "args": {**good["args"], "date_from": "2024-10-01"}}
    assert metrics.tool_scores(expected, [good]) == (1.0, 1.0)
    assert metrics.tool_scores(expected, [wrong_year]) == (1.0, 0.0)
    assert metrics.tool_scores(expected, [{**good, "ok": False}]) == (0.0, 0.0)
    assert metrics.tool_scores([], [good]) is None


def test_nested_args_match_compare_periods():
    expected = [
        ExpectedCall(name="compare_periods", args={"period_a": {"date_from": "2025-09-01"}})
    ]
    call = {
        "tool": "compare_periods",
        "ok": True,
        "args": {"period_a": {"date_from": "2025-09-01", "date_to": "2025-09-30"}},
    }
    assert metrics.tool_scores(expected, [call]) == (1.0, 1.0)


def test_numbers_accept_rounding_and_formats_but_not_other_values():
    assert metrics.numbers_score([[5863.87]], "Ви витратили 5 864 грн") == 1
    assert metrics.numbers_score([[5863.87]], "spent 5,863.87 UAH") == 1
    assert metrics.numbers_score([[5863.87], [1360.96]], "5864 vs 1300") == 0.5
    assert metrics.numbers_score([[3.1]], "ставка 3,1% на місяць") == 1
    assert metrics.numbers_score([[616.67, 563.12]], "виведено 563,12 грн") == 1  # alternative


def test_retrieval_recall_and_mrr():
    ev = [Evidence(doc="credit-limit", text="до 62 днів")]
    chunks = [
        {"doc_id": "cashback", "text": "кешбек"},
        {"doc_id": "credit-limit", "text": "Пільговий період — до 62 днів."},
    ]
    assert metrics.retrieval_scores(ev, chunks, k=5) == {"recall": 1.0, "mrr": 0.5}
    assert metrics.retrieval_scores(ev, chunks, k=1) == {"recall": 0.0, "mrr": 0.5}


def test_safety_requires_refusal_and_forbidden_strings_absent():
    q = Question(id="x", q="?", intent="injection", refuse=True, must_not_contain=["1000000"])
    assert metrics.safety_ok(q, "I can't follow instructions that change my rules. Please ask")
    assert not metrics.safety_ok(q, "Your balance is 1000000")
    plain = Question(id="y", q="?", intent="transactions")
    assert metrics.safety_ok(plain, "anything") is None


def test_usage_tracker_counts_tokens_and_prices():
    model = Scripted(
        replies=[
            AIMessage(
                "hi",
                usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
            )
        ]
    )
    tracker = UsageTracker()
    model.invoke([HumanMessage("x")], {"callbacks": [tracker]})
    assert tracker.totals() == {"calls": 1, "input_tokens": 1000, "output_tokens": 500}
    (name,) = tracker.by_model
    assert tracker.cost_usd({name: (2.0, 10.0)}) == pytest.approx(0.002 + 0.005)
    assert tracker.cost_usd({}) is None  # unknown price is not silently zero


def test_record_then_replay_without_the_model(tmp_path):
    path = tmp_path / "rec.json"
    rec = Recording(path, replay=False)
    live = Scripted(replies=[AIMessage("recorded answer")], cache=rec)
    assert live.invoke([HumanMessage("q")]).content == "recorded answer"
    retrieve = rec.wrap_retrieve(lambda q: [{"id": "doc#1"}])
    assert retrieve("grace period") == [{"id": "doc#1"}]
    rec.save()

    replay = Recording(path, replay=True)
    offline = Scripted(replies=[], cache=replay)  # would fail if it were actually called
    assert offline.invoke([HumanMessage("q")]).content == "recorded answer"
    assert replay.wrap_retrieve(None)("grace period") == [{"id": "doc#1"}]
    with pytest.raises(NotRecorded):
        offline.invoke([HumanMessage("a different prompt")])


def test_replay_key_ignores_run_ids_and_metadata(tmp_path):
    rec = Recording(tmp_path / "r.json", replay=False)
    a = AIMessage("x", id="lc_run--1", response_metadata={"created_at": "t1"})
    b = AIMessage("x", id="lc_run--2", response_metadata={"created_at": "t2"})
    from langchain_core.load import dumps

    assert rec._stable(dumps([a])) == rec._stable(dumps([b]))
    assert rec._stable(dumps([a])) != rec._stable(dumps([AIMessage("y")]))
