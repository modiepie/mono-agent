from datetime import datetime

from mono_agent.config import Settings
from mono_agent.mono.cache import Cache
from mono_agent.mono.client import MonoClient
from mono_agent.mono.factory import build_client
from mono_agent.mono.ratelimit import Unlimited
from mono_agent.synthetic.fake_api import FakeMonoApi
from mono_agent.synthetic.generator import KYIV, generate

JAN = int(datetime(2025, 1, 1, tzinfo=KYIV).timestamp())
JUL = int(datetime(2025, 7, 1, tzinfo=KYIV).timestamp())
DEC = int(datetime(2025, 12, 31, 23, 59, tzinfo=KYIV).timestamp())
NOW = DEC + 86_400


DATA = generate(42)


def synthetic() -> tuple[MonoClient, Cache, FakeMonoApi]:
    api = FakeMonoApi(DATA.client_info, DATA.statements, DATA.rates)
    buckets = {k: Unlimited() for k in ("client-info", "statement", "currency")}
    return MonoClient("t", transport=api.transport, buckets=buckets), Cache(":memory:"), api


def test_factory_builds_working_synthetic_client():
    client = build_client(Settings(_env_file=None, mono_token=None))  # type: ignore[call-arg]
    assert client.client_info() == DATA.client_info


def test_repeated_sync_never_hits_api_again():
    client, cache, api = synthetic()
    account = cache.client_info(client, now=NOW).accounts[0].id
    first = cache.sync_statement(client, account, JAN, DEC, now=NOW)
    calls = len(api.calls)
    assert first > 500
    assert cache.sync_statement(client, account, JAN, DEC, now=NOW) == 0
    assert cache.sync_statement(client, account, JUL, DEC, now=NOW) == 0
    assert len(api.calls) == calls


def test_only_missing_part_is_fetched():
    client, cache, api = synthetic()
    account = cache.client_info(client, now=NOW).accounts[0].id
    cache.sync_statement(client, account, JUL, DEC, now=NOW)
    before = len(api.calls)
    cache.sync_statement(client, account, JAN, DEC, now=NOW)
    paths = api.calls[before:]
    assert all(int(p.split("/")[-1]) < JUL for p in paths)
    total = cache.db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert total == len(client.statement(account, JAN, DEC))


def test_future_is_not_marked_as_covered():
    client, cache, _ = synthetic()
    account = cache.client_info(client, now=NOW).accounts[0].id
    cache.sync_statement(client, account, JAN, DEC, now=JUL)
    assert max(end for _, end in cache.covered(account)) == JUL


def test_client_info_is_cached_with_ttl_and_masked():
    client, cache, api = synthetic()
    cache.client_info(client, now=NOW)
    cache.client_info(client, now=NOW + 60)
    assert len(api.calls) == 1
    cache.client_info(client, now=NOW + 3600)
    assert len(api.calls) == 2
    for (iban,) in cache.db.execute("SELECT iban FROM accounts"):
        assert iban.startswith("UA****") and len(iban) == 10


def test_stored_text_is_masked():
    client, cache, _ = synthetic()
    account = cache.client_info(client, now=NOW).accounts[0].id
    cache.sync_statement(client, account, JAN, DEC, now=NOW)
    dump = " ".join(
        str(v) for row in cache.db.execute("SELECT * FROM transactions") for v in tuple(row)
    )
    assert "4441 1144 5555 6789" not in dump
    assert "UA213223130000026007233566001" not in dump
    assert "****6789" in dump
