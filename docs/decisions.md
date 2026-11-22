# Design decisions

Each entry: what, why, alternatives considered, trade-offs. Added as decisions are made.

## D1. uv + Python 3.12, `src/` layout

- **What:** uv manages Python, venv and lockfile. Code lives in `src/mono_agent`.
- **Why:** uv is one fast tool for interpreter, deps and lock (`uv.lock` makes CI reproducible). 3.12 is the newest version every ML dependency (torch, sentence-transformers, bitsandbytes) supports well. The `src/` layout forces tests to import the installed package, which catches packaging bugs.
- **Alternatives:** Poetry (slower, no Python management), pip + requirements.txt (no lockfile by default), flat layout (simpler, but tests can accidentally import local files).
- **Trade-offs:** uv is newer than Poetry; fewer people have used it in older teams.

## D2. Dependencies added per milestone

- **What:** `pyproject.toml` starts with only httpx and pydantic; LangGraph, Qdrant client etc. arrive with the code that uses them.
- **Why:** every dependency in the file has a reason I can point to in the code. Smaller install while early milestones are built.
- **Trade-offs:** pyproject changes in several commits instead of one.

## D3. Config via pydantic-settings, token as `SecretStr`

- **What:** one `Settings` class reads env/.env. `MONO_TOKEN` is a `SecretStr`; its absence means synthetic mode.
- **Why:** typed config validated at startup; `SecretStr` hides the token in `repr()`, logs and tracebacks, so it can't leak by accident. One switch (token set or not) decides real vs synthetic, nothing else to configure.
- **Alternatives:** `os.environ` reads scattered across modules (untyped, easy to log by mistake).
- **Trade-offs:** must call `.get_secret_value()` explicitly when building the HTTP header; that is the point.

## D4. SQLite for cache, query layer and checkpointer

- **What:** one SQLite file holds cached statements; tools query it; LangGraph's SQLite checkpointer stores threads.
- **Why:** a single user's statements are thousands of rows, well inside SQLite's comfort zone. Zero ops, works in CI, and the tools can use real SQL aggregation instead of Python loops.
- **Alternatives:** Postgres (needed for multi-user/concurrent writes; overkill here), DuckDB (faster analytics, but a second engine to explain for no measurable gain at this size).
- **Trade-offs:** not suited for many concurrent writers; the migration path to Postgres is the same SQL plus `langgraph-checkpoint-postgres`.

## D5. Synchronous HTTP client (httpx), not async

- **What:** `MonoClient` uses `httpx.Client`.
- **Why:** the API allows one statement call per 60 s, so concurrency buys nothing: requests are serialized by the bank anyway. Sync code is easier to read, test and step through. LangGraph runs sync tools fine, and FastAPI runs sync code in a threadpool.
- **Alternatives:** `httpx.AsyncClient` everywhere.
- **Trade-offs:** if the service ever served many users at once, each blocked sync call would hold a worker thread. The switch to async would be mechanical (same httpx API).

## D6. Token-bucket rate limiter, one bucket per endpoint

- **What:** `TokenBucket(capacity=1, period=60)` for client-info, statement and currency, checked before every request (including retries).
- **Why:** the bank's limit is "1 call per 60 s"; a bucket of capacity 1 enforces exactly that and waits only as long as needed (if the last call was 45 s ago, it waits 15 s). Clock and sleep are injected, so tests run instantly with a fake clock.
- **Alternatives:** a plain `sleep(60)` between calls (wastes time), reacting only to 429 (the bank would see us violate the limit on purpose), a library like `ratelimit` (extra dependency for ~30 lines).
- **Trade-offs:** the lock is held while sleeping, so concurrent callers queue in order. That is intended. The limit lives in process memory: two processes with the same token would not coordinate.

## D7. Retries: 429 / 5xx / network errors, exponential backoff, no jitter

- **What:** up to 5 retries with delays 2, 4, 8, 16, 32 s (capped at 60), or `Retry-After` if larger. 4xx other than 429 fail immediately with `MonoApiError`.
- **Why:** 429 and 5xx are transient; 400/401/403 are our mistakes and retrying can't fix them. Backoff gives the server room.
- **Alternatives:** `tenacity` (fine library, but the loop is short and explicit here), jitter.
- **Trade-offs:** no jitter, because jitter exists to stop many clients retrying in lockstep, and there is one client. Error messages include the path and response body but never headers, so the token can't leak (there is a test for this).

## D8. 31-day windows and 500-item paging inside the client

- **What:** `statement(account, start, end)` splits any range into windows of at most 2,682,000 s. If a response has exactly 500 items, it requests again with `to` = the oldest item's time. Items are de-duplicated by id.
- **Why:** callers (tools, cache) shouldn't need to know API limits. Paging by `to` is what the bank documents. Items that share the boundary second come back twice, hence the de-duplication by id.
- **Edge case:** if more than 500 items share one second, moving `to` doesn't make progress; the client steps back 1 s and those extra items are lost. This is an API limitation, documented in code with a `shortcut:` comment. It cannot happen for a personal account.

## D9. SQLite cache with explicit "covered ranges"

- **What:** fetched items go into `transactions`; each fetched interval is recorded in `synced_ranges`. A sync computes `requested − covered` and fetches only the gaps. Client info (TTL 10 min) and currency rates (TTL 5 min) are cached as JSON in a `kv` table.
- **Why:** "repeated questions never hit the API" needs to know what was fetched, not just what exists: a month with zero transactions is still "fetched". Storing intervals (not just a max timestamp) also handles out-of-order requests (ask about July, then January).
- **Detail:** a range is only marked covered up to *now*, because future transactions can still appear. Rows are upserted (`REPLACE`), so a re-fetched item that went from hold to settled overwrites the old one.
- **Trade-offs / known gap:** items on hold inside an already-covered range are not re-checked. Fix when needed: always re-sync the last few days.

## D10. Mask PANs and IBANs at ingestion

- **What:** `mask_text` replaces 13–19-digit card numbers with `****1234` and Ukrainian IBANs with `UA****1234`. It runs before data enters the cache, so everything downstream (tools, LLM prompts, traces) only ever sees masked values.
- **Why:** masking at the single entry point is much safer than remembering to mask at every exit (prompt, log, trace, API response). There is nothing in the app that needs the full number.
- **Alternatives:** masking only before the LLM call (easy to miss a path, e.g. Langfuse traces of tool outputs), Presidio-style NER (heavy, and the formats here are regular).
- **Trade-offs:** a regex can mask a long non-card number (e.g. a 16-digit order id); that's an acceptable false positive. No Luhn check, on purpose: masking a non-card is harmless, missing a card is not.

## D11. Synthetic data: same API shape, served through a fake HTTP API

- **What:** `generate(seed)` produces `ClientInfo`, `StatementItem`s and `CurrencyRate`s for calendar year 2025: 3 accounts (UAH black card with credit limit, USD, EUR), 2 jars, about 900 items. `FakeMonoApi` serves them through `httpx.MockTransport` and enforces the real limits (31-day window, newest 500, token required).
- **Why:** synthetic mode exercises the exact same client, paging and cache code as real mode. The only branch on mode is `mono/factory.py`. A fixed year (not "last 12 months") makes eval answers stable forever. `random.Random(seed)` with no set/hash iteration makes output identical across runs and machines.
- **Realism:** real Ukrainian merchant brands and MCC codes; fictional people and employer. Salary with a mid-year raise, rent, utilities with winter heating, subscriptions in UAH and USD, FX purchases, a trip paid in PLN from the EUR account, cashback on chosen categories with monthly payout, P2P transfers, last two days on hold.
- **Planted stories for the eval:** October taxi spike explained by a car-service visit (no fuel purchases Oct 3–17) plus a 15% tariff rise; overlapping music subscriptions; MEGOGO cancelled; Netflix price increase; duplicate charge + refund; a return; an outlier laptop purchase; unknown foreign PayPal charges; prompt-injection text in a merchant name and in a transfer comment (with a card number and IBAN that must be masked).
- **Trade-offs:** hand-written behaviour, not learned from real data, so distributions are plausible rather than statistically faithful. Fine for testing an agent, not for training a fraud model.

## D12. Ruff rules RUF001–003 off

- **What:** ruff's "ambiguous Unicode character" rules are disabled.
- **Why:** they flag Cyrillic letters that look like Latin ones (`С`, `О`, `а`). In a Ukrainian project those are intentional, and the rule would fire on every merchant name.
