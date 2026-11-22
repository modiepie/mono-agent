"""In-process fake of the monobank Personal API, served through httpx.MockTransport.

Enforces the same constraints as the real API (31-day + 1 h window, newest 500 items, token
required), so synthetic mode runs the exact client code path used with the real bank.
"""

import json
import re

import httpx

from mono_agent.mono.models import ClientInfo, CurrencyRate, StatementItem
from mono_agent.mono.ranges import MAX_WINDOW_S

_STATEMENT = re.compile(r"^/personal/statement/([^/]+)/(\d+)(?:/(\d+))?$")


class FakeMonoApi:
    def __init__(
        self,
        client_info: ClientInfo,
        statements: dict[str, list[StatementItem]],
        rates: list[CurrencyRate] | None = None,
        page_limit: int = 500,
    ) -> None:
        self.client_info = client_info
        self.statements = statements
        self.rates = rates or []
        self.page_limit = page_limit
        self.calls: list[str] = []  # request paths, for tests
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if path == "/bank/currency":
            return httpx.Response(200, json=[r.to_api() for r in self.rates])
        if not request.headers.get("X-Token"):
            return _error(403, "Unknown 'X-Token'")
        if path == "/personal/client-info":
            return httpx.Response(200, json=self.client_info.to_api())
        if m := _STATEMENT.match(path):
            account, start = m.group(1), int(m.group(2))
            end = int(m.group(3)) if m.group(3) else start + MAX_WINDOW_S
            if account == "0":
                account = self.client_info.accounts[0].id
            if account not in self.statements:
                return _error(400, "invalid account")
            if end - start > MAX_WINDOW_S:
                return _error(400, "Period must be no more than 31 days")
            items = [i for i in self.statements[account] if start <= i.time <= end]
            items.sort(key=lambda i: i.time, reverse=True)
            return httpx.Response(200, json=[i.to_api() for i in items[: self.page_limit]])
        return _error(404, "not found")


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, content=json.dumps({"errorDescription": message}))
