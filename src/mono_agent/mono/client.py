"""HTTP client for the monobank Personal API.

Rate limits (1 call / 60 s per endpoint), the 31-day window, 500-item paging and retries all
live here, so callers only ever ask for "the statement between A and B".
"""

import logging
import time
from collections.abc import Callable, Mapping

import httpx
from pydantic import TypeAdapter

from mono_agent.mono.models import ClientInfo, CurrencyRate, StatementItem
from mono_agent.mono.ranges import split_window
from mono_agent.mono.ratelimit import TokenBucket

log = logging.getLogger(__name__)

PAGE_LIMIT = 500  # the API returns at most this many items, newest first
RETRY_STATUSES = {429, 500, 502, 503, 504}

_items = TypeAdapter(list[StatementItem])
_rates = TypeAdapter(list[CurrencyRate])


class MonoApiError(RuntimeError):
    def __init__(self, status: int, path: str, body: str) -> None:
        # Path and body only: the token is in a header and never goes into the message.
        super().__init__(f"monobank API {status} on {path}: {body[:200]}")
        self.status = status


class MonoClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.monobank.ua",
        transport: httpx.BaseTransport | None = None,
        buckets: Mapping[str, TokenBucket] | None = None,
        max_retries: int = 5,
        timeout: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"X-Token": token},
            transport=transport,
            timeout=timeout,
        )
        self._buckets = buckets or {
            "client-info": TokenBucket(),
            "statement": TokenBucket(),
            "currency": TokenBucket(),
        }
        self._max_retries = max_retries
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def _get(self, path: str, bucket: str) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._buckets[bucket].acquire()
            retry_after = None
            try:
                resp = self._http.get(path)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise
                log.warning("transport error on %s: %s", path, type(exc).__name__)
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                    raise MonoApiError(resp.status_code, path, resp.text)
                log.warning("HTTP %s on %s, attempt %s", resp.status_code, path, attempt + 1)
                retry_after = resp.headers.get("Retry-After")
            # Exponential backoff 2, 4, 8, ... s. No jitter: a single client, no herd to spread.
            delay = min(2.0 ** (attempt + 1), 60.0)
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
            self._sleep(delay)
        raise AssertionError("unreachable")

    def client_info(self) -> ClientInfo:
        return ClientInfo.model_validate_json(
            self._get("/personal/client-info", "client-info").content
        )

    def currency_rates(self) -> list[CurrencyRate]:
        return _rates.validate_json(self._get("/bank/currency", "currency").content)

    def statement(self, account: str, start: int, end: int) -> list[StatementItem]:
        """All items in [start, end], newest first, de-duplicated by id."""
        seen: dict[str, StatementItem] = {}
        for w_start, w_end in reversed(split_window(start, end)):
            to = w_end
            while True:
                path = f"/personal/statement/{account}/{w_start}/{to}"
                page = _items.validate_json(self._get(path, "statement").content)
                for item in page:
                    seen.setdefault(item.id, item)
                if len(page) < PAGE_LIMIT:
                    break
                # Full page: older items may remain. Move `to` back to the oldest item seen;
                # items sharing that second come back again and are dropped by id.
                oldest = page[-1].time
                # shortcut: >500 items in one second would loop; step back 1 s and lose them.
                to = oldest if oldest < to else to - 1
                if to < w_start:
                    break
        return sorted(seen.values(), key=lambda i: i.time, reverse=True)
