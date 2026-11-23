import json

from eval.dataset import load
from finetune.router_data import INTENTS, MAX_SIMILARITY, OUT, build, similarity, to_chat
from finetune.router_eval import parse


def test_training_data_is_balanced_and_held_out():
    tests = [q.q for q in load()]
    rows = build(tests)
    assert build(tests) == rows  # deterministic
    counts = {i: sum(1 for _, x in rows if x == i) for i in INTENTS}
    assert min(counts.values()) >= 100, counts
    # The eval is the test set: no training question may be worded like an eval question.
    assert max(similarity(q, t) for q, _ in rows for t in tests) < MAX_SIMILARITY


def test_committed_data_matches_the_generator():
    rows = build([q.q for q in load()])
    committed = [
        json.loads(line)
        for name in ("valid", "train")
        for line in (OUT / f"{name}.jsonl").read_text().splitlines()
    ]
    assert committed == [to_chat(q, i) for q, i in rows]


def test_parse_takes_the_first_label():
    assert parse("Transactions.") == "transactions"
    assert parse("out of scope") == "out_of_scope"
    assert parse("knowledge, not mixed") == "knowledge"
    assert parse("I think") == "invalid"
