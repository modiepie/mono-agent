"""SQLite cache of accounts and statements, plus the ranges already fetched.

Everything stored here is already masked, so nothing read from the cache can leak a PAN/IBAN.
The transactions table is also what the agent's query tools read (milestone 2).
"""

import json
import sqlite3
import time
from pathlib import Path

from mono_agent.masking import mask_text
from mono_agent.mono.client import MonoClient
from mono_agent.mono.models import ClientInfo, CurrencyRate, StatementItem
from mono_agent.mono.ranges import Range, subtract

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY, type TEXT, currency_code INTEGER, balance INTEGER,
    credit_limit INTEGER, cashback_type TEXT, masked_pan TEXT, iban TEXT
);
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY, account_id TEXT NOT NULL, time INTEGER NOT NULL,
    description TEXT, mcc INTEGER, original_mcc INTEGER, hold INTEGER,
    amount INTEGER, operation_amount INTEGER, currency_code INTEGER,
    commission_rate INTEGER, cashback_amount INTEGER, balance INTEGER,
    comment TEXT, counter_name TEXT, counter_iban TEXT, counter_edrpou TEXT
);
CREATE INDEX IF NOT EXISTS ix_tx_account_time ON transactions(account_id, time);
CREATE TABLE IF NOT EXISTS synced_ranges (
    account_id TEXT NOT NULL, start INTEGER NOT NULL, "end" INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, fetched_at INTEGER, value TEXT);
"""


class Cache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # --- client info / rates: small JSON blobs with a TTL --------------------------------

    def _get_kv(self, key: str, max_age: float, now: float) -> str | None:
        row = self.db.execute("SELECT fetched_at, value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row and now - row["fetched_at"] <= max_age else None

    def _put_kv(self, key: str, value: str, now: float) -> None:
        with self.db:
            self.db.execute("REPLACE INTO kv VALUES (?, ?, ?)", (key, int(now), value))

    def client_info(self, client: MonoClient, max_age: float = 600, now: float | None = None):
        now = time.time() if now is None else now
        if cached := self._get_kv("client_info", max_age, now):
            return ClientInfo.model_validate_json(cached)
        info = client.client_info()
        masked = info.model_copy(
            update={
                "accounts": [
                    a.model_copy(update={"iban": mask_text(a.iban) or ""}) for a in info.accounts
                ]
            }
        )
        self._put_kv("client_info", masked.model_dump_json(), now)
        with self.db:
            self.db.executemany(
                "REPLACE INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        a.id,
                        a.type,
                        a.currency_code,
                        a.balance,
                        a.credit_limit,
                        a.cashback_type,
                        ",".join(a.masked_pan),
                        a.iban,
                    )
                    for a in masked.accounts
                ],
            )
        return masked

    def currency_rates(
        self, client: MonoClient, max_age: float = 300, now: float | None = None
    ) -> list[CurrencyRate]:
        now = time.time() if now is None else now
        if cached := self._get_kv("currency", max_age, now):
            return [CurrencyRate.model_validate(r) for r in json.loads(cached)]
        rates = client.currency_rates()
        self._put_kv("currency", json.dumps([r.model_dump() for r in rates]), now)
        return rates

    # --- statements ----------------------------------------------------------------------

    def covered(self, account: str) -> list[Range]:
        rows = self.db.execute(
            'SELECT start, "end" FROM synced_ranges WHERE account_id = ?', (account,)
        )
        return [(r[0], r[1]) for r in rows]

    def store_items(self, account: str, items: list[StatementItem]) -> None:
        rows = [
            (
                i.id,
                account,
                i.time,
                mask_text(i.description),
                i.mcc,
                i.original_mcc,
                int(i.hold),
                i.amount,
                i.operation_amount,
                i.currency_code,
                i.commission_rate,
                i.cashback_amount,
                i.balance,
                mask_text(i.comment),
                mask_text(i.counter_name),
                mask_text(i.counter_iban),
                i.counter_edrpou,
            )
            for i in items
        ]
        with self.db:
            # REPLACE so a re-fetched item (e.g. hold -> settled) overwrites the old row.
            self.db.executemany(f"REPLACE INTO transactions VALUES ({','.join('?' * 17)})", rows)

    def sync_statement(
        self, client: MonoClient, account: str, start: int, end: int, now: float | None = None
    ) -> int:
        """Fetch only the parts of [start, end] not cached yet. Returns API items received."""
        now = time.time() if now is None else now
        # Never mark the future as covered: new transactions can still arrive there.
        # shortcut: items still on hold inside a covered range are not re-checked; add a
        # "re-sync last N days" pass if stale holds become a problem with real data.
        end_covered = min(end, int(now))
        fetched = 0
        for gap_start, gap_end in subtract((start, end_covered), self.covered(account)):
            items = client.statement(account, gap_start, gap_end)
            self.store_items(account, items)
            fetched += len(items)
            with self.db:
                self.db.execute(
                    "INSERT INTO synced_ranges VALUES (?, ?, ?)", (account, gap_start, gap_end)
                )
        return fetched
