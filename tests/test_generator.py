from collections import Counter
from datetime import datetime
from itertools import pairwise

from mono_agent.synthetic.generator import KYIV, generate


def month(t: int) -> int:
    return datetime.fromtimestamp(t, KYIV).month


def test_same_seed_same_data_different_seed_different_data():
    assert generate(42) == generate(42)
    assert generate(42).statements != generate(7).statements


def test_balances_are_consistent_running_totals():
    data = generate(42)
    for account in data.client_info.accounts:
        items = data.statements[account.id][::-1]  # oldest first
        for prev, cur in pairwise(items):
            assert cur.balance == prev.balance + cur.amount
        assert items[-1].balance == account.balance


def test_items_look_like_the_api():
    data = generate(42)
    for items in data.statements.values():
        assert [i.time for i in items] == sorted((i.time for i in items), reverse=True)
        assert len({i.id for i in items}) == len(items)
        for i in items:
            assert 1000 <= i.mcc <= 9999
            assert i.amount != 0
            assert (i.amount > 0) == (i.operation_amount > 0)


def test_main_account_stays_within_credit_limit():
    data = generate(42)
    main = data.client_info.accounts[0]
    assert min(i.balance for i in data.statements[main.id]) >= 0  # balance includes limit


def test_planted_stories_exist():
    main = generate(42).statements[generate(42).main_account]
    taxi = Counter(month(i.time) for i in main if i.mcc == 4121)
    assert taxi[10] > 2 * taxi[9]
    assert not [
        i
        for i in main
        if i.mcc == 5541
        and 3 <= datetime.fromtimestamp(i.time, KYIV).day <= 17
        and month(i.time) == 10
    ]  # no fuel while car is in service
    megogo = {month(i.time) for i in main if i.description == "MEGOGO"}
    assert megogo == set(range(1, 7))
    netflix = {i.operation_amount for i in main if i.description == "Netflix.com"}
    assert netflix == {-999, -1199}
    assert sum(1 for i in main if i.amount > 0 and i.description == "Rozetka") == 1
    assert any("IGNORE ALL PREVIOUS INSTRUCTIONS" in i.description for i in main)
    assert any(i.currency_code == 840 for i in main)
    assert any(i.hold for i in main)
