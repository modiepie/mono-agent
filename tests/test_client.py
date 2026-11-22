import logging

import httpx
import pytest

from mono_agent.mono.client import MonoApiError, MonoClient
from mono_agent.mono.models import Account, ClientInfo, StatementItem
from mono_agent.mono.ranges import MAX_WINDOW_S
from mono_agent.mono.ratelimit import Unlimited
from mono_agent.synthetic.fake_api import FakeMonoApi

DAY = 86_400
T0 = 1_735_689_600  # 2025-01-01 UTC


def item(i: int, t: int) -> StatementItem:
    return StatementItem(
        id=f"tx{i}",
        time=t,
        description="Сільпо",
        mcc=5411,
        original_mcc=5411,
        hold=False,
        amount=-100,
        operation_amount=-100,
        currency_code=980,
        balance=0,
    )


def client_for(transport: httpx.BaseTransport, sleeps: list[float] | None = None) -> MonoClient:
    buckets = {k: Unlimited() for k in ("client-info", "statement", "currency")}
    sleep = sleeps.append if sleeps is not None else (lambda _s: None)
    return MonoClient("secret-token", transport=transport, buckets=buckets, sleep=sleep)


def fake_api(items: list[StatementItem]) -> FakeMonoApi:
    info = ClientInfo(
        client_id="c",
        name="n",
        accounts=[Account(id="acc", balance=0, type="black", currency_code=980)],
    )
    return FakeMonoApi(info, {"acc": sorted(items, key=lambda i: -i.time)})


def test_long_range_is_split_into_31_day_windows():
    items = [item(i, T0 + i * DAY) for i in range(100)]
    api = fake_api(items)
    got = client_for(api.transport).statement("acc", T0, T0 + 99 * DAY)
    assert {i.id for i in got} == {i.id for i in items}
    assert len(api.calls) == 4  # 100 days / 31 days -> 4 windows
    assert [i.time for i in got] == sorted((i.time for i in got), reverse=True)


def test_full_page_triggers_paging_backwards():
    # 1200 items inside one window, several sharing the same second at page boundaries.
    items = [item(i, T0 + i // 3) for i in range(1200)]
    api = fake_api(items)
    got = client_for(api.transport).statement("acc", T0, T0 + DAY)
    assert len(got) == 1200
    assert len({i.id for i in got}) == 1200  # de-duplicated across page boundaries
    assert len(api.calls) == 3


def test_exactly_500_items_makes_one_more_call_then_stops():
    items = [item(i, T0 + i) for i in range(500)]
    api = fake_api(items)
    assert len(client_for(api.transport).statement("acc", T0, T0 + DAY)) == 500
    assert len(api.calls) == 2


def test_window_never_exceeds_api_limit():
    api = fake_api([])
    client_for(api.transport).statement("acc", T0, T0 + 365 * DAY)
    for path in api.calls:
        _, _, _, _, start, end = path.split("/")
        assert int(end) - int(start) <= MAX_WINDOW_S


def test_retries_429_and_5xx_with_backoff():
    responses = iter([httpx.Response(429), httpx.Response(503), httpx.Response(200, json=[])])
    sleeps: list[float] = []
    client = client_for(httpx.MockTransport(lambda _r: next(responses)), sleeps)
    assert client.statement("acc", T0, T0 + 10) == []
    assert sleeps == [2.0, 4.0]


def test_retry_after_header_is_respected():
    responses = iter(
        [httpx.Response(429, headers={"Retry-After": "61"}), httpx.Response(200, json=[])]
    )
    sleeps: list[float] = []
    client_for(httpx.MockTransport(lambda _r: next(responses)), sleeps).statement("a", 0, 1)
    assert sleeps == [61.0]


def test_gives_up_after_max_retries():
    client = client_for(httpx.MockTransport(lambda _r: httpx.Response(500)))
    with pytest.raises(MonoApiError) as exc:
        client.statement("acc", T0, T0 + 10)
    assert exc.value.status == 500


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"errorDescription": "bad"})

    with pytest.raises(MonoApiError):
        client_for(httpx.MockTransport(handler)).statement("acc", T0, T0 + 10)
    assert len(calls) == 1


def test_token_is_sent_but_never_logged_or_in_errors(caplog):
    seen = []

    def handler(request):
        seen.append(request.headers["X-Token"])
        return httpx.Response(503, text="down")

    caplog.set_level(logging.DEBUG)
    with pytest.raises(MonoApiError) as exc:
        client_for(httpx.MockTransport(handler)).client_info()
    assert seen[0] == "secret-token"
    assert "secret-token" not in caplog.text
    assert "secret-token" not in str(exc.value)
