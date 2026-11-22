"""Pydantic models mirroring the monobank Personal API JSON (camelCase on the wire).

Amounts are integers in minor units (kopiykas / cents). Currency codes are ISO 4217 numeric.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    def to_api(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)


class Account(ApiModel):
    id: str
    send_id: str = ""
    balance: int  # includes credit_limit, as the real API reports it
    credit_limit: int = 0
    type: str  # black, white, platinum, iron, fop, yellow, eAid
    currency_code: int
    cashback_type: str | None = None  # None, UAH, Miles
    masked_pan: list[str] = []
    iban: str = ""


class Jar(ApiModel):
    id: str
    send_id: str = ""
    title: str
    description: str = ""
    currency_code: int
    balance: int
    goal: int | None = None


class ClientInfo(ApiModel):
    client_id: str
    name: str
    web_hook_url: str = ""
    permissions: str = ""
    accounts: list[Account]
    jars: list[Jar] = []


class StatementItem(ApiModel):
    id: str
    time: int  # unix seconds
    description: str
    mcc: int
    original_mcc: int
    hold: bool
    amount: int  # account currency, negative = debit
    operation_amount: int  # operation currency
    currency_code: int  # operation currency
    commission_rate: int = 0
    cashback_amount: int = 0
    balance: int  # account balance after the operation
    comment: str | None = None
    receipt_id: str | None = None
    invoice_id: str | None = None
    counter_edrpou: str | None = None
    counter_iban: str | None = None
    counter_name: str | None = None


class CurrencyRate(ApiModel):
    currency_code_a: int
    currency_code_b: int
    date: int
    rate_sell: float | None = None
    rate_buy: float | None = None
    rate_cross: float | None = None
