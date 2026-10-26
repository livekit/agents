from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

RECORDS = json.loads((Path(__file__).parent / "cardholders.json").read_text())

BANK_NAME = RECORDS["bank"]

AuthState = Literal["unauthenticated", "authenticating", "authenticated"]
CardState = Literal["active", "locked"]


class BankError(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    name: str
    date_of_birth: str
    last_four: str
    merchant: str
    amount: str

    def matches(self, other: Identity) -> bool:
        return (
            _canonical(self.name) == _canonical(other.name)
            and self.date_of_birth.strip() == other.date_of_birth.strip()
            and _digits(self.last_four) == _digits(other.last_four)
            and _canonical(self.merchant) == _canonical(other.merchant)
            and _cents(self.amount) is not None
            and _cents(self.amount) == _cents(other.amount)
        )

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name,
            "dateOfBirth": self.date_of_birth,
            "lastFour": self.last_four,
            "merchant": self.merchant,
            "amount": self.amount,
        }


ON_FILE = [
    Identity(
        name=c["name"],
        date_of_birth=c["dob"],
        last_four=c["last_four"],
        merchant=c["last_purchase"]["merchant"],
        amount=c["last_purchase"]["amount"],
    )
    for c in RECORDS["cardholders"]
]


@dataclass(frozen=True)
class FlaggedPurchase:
    merchant: str
    place: str
    country: str
    amount: str

    def to_json(self) -> dict[str, str]:
        return {
            "merchant": self.merchant,
            "place": self.place,
            "country": self.country,
            "amount": self.amount,
        }


FLAGGED = FlaggedPurchase(
    merchant="Trattoria da Enzo", place="Rome, Italy", country="Italy", amount="186.40"
)
LOSS_PREVENTION_CASE = "LP-20417"
LONGEST_TRAVEL_NOTICE_DAYS = 90


@dataclass
class TravelNotice:
    destination: str
    until: date


@dataclass
class Alert:
    """An outbound call about a flagged purchase, to the cardholder it was made on."""

    cardholder: Identity
    purchase: FlaggedPurchase = FLAGGED
    checks: list[bool | None] = field(default_factory=list)
    made_by_cardholder: bool | None = None
    card: CardState = "active"
    case_id: str | None = None
    travel: TravelNotice | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "cardholder": self.cardholder.name,
            "lastFour": self.cardholder.last_four,
            "purchase": self.purchase.to_json(),
            "checks": self.checks,
            "madeByCardholder": self.made_by_cardholder,
            "card": self.card,
            "caseId": self.case_id,
            "travel": None
            if self.travel is None
            else {
                "destination": self.travel.destination,
                "until": self.travel.until.isoformat(),
            },
        }


Publish = Callable[[str, Any], Awaitable[None]]


async def _discard(_key: str, _value: Any) -> None:
    return None


@dataclass
class Bank:
    publish: Publish = _discard
    lookup_seconds: float = 1.5
    attempts: list[dict[str, Any]] = field(default_factory=list)
    state: AuthState = "unauthenticated"
    alert: Alert | None = None

    async def authenticate(self, claimed: Identity) -> Identity | None:
        attempt: dict[str, Any] = {
            "submitted": claimed.to_json(),
            "authenticated": None,
        }
        self.attempts.append(attempt)
        await self.publish("attempts", self.attempts)
        await self._set_state("authenticating")
        await asyncio.sleep(self.lookup_seconds)
        match = next((c for c in ON_FILE if c.matches(claimed)), None)
        attempt["authenticated"] = match is not None
        await self.publish("attempts", self.attempts)
        await self._set_state("unauthenticated" if match is None else "authenticated")
        return match

    async def open_alert(self) -> Alert:
        self.alert = Alert(cardholder=ON_FILE[0])
        await self._publish_alert()
        return self.alert

    async def confirm_birth_date(self, date_of_birth: str) -> bool:
        alert = self._require_alert()
        alert.checks.append(None)
        await self._publish_alert()
        await self._set_state("authenticating")
        await asyncio.sleep(self.lookup_seconds)
        verified = date_of_birth.strip() == alert.cardholder.date_of_birth
        alert.checks[-1] = verified
        await self._publish_alert()
        await self._set_state("authenticated" if verified else "unauthenticated")
        return verified

    async def record_answer(self, made_by_cardholder: bool) -> Alert:
        alert = self._require_verified()
        alert.made_by_cardholder = made_by_cardholder
        if made_by_cardholder:
            alert.card = "active"
            alert.case_id = None
        else:
            alert.card = "locked"
            alert.case_id = LOSS_PREVENTION_CASE
            alert.travel = None
        await self._publish_alert()
        return alert

    async def set_travel_notice(
        self, destination: str, until: date, today: date
    ) -> Alert:
        alert = self._require_verified()
        if alert.made_by_cardholder is not True:
            raise BankError(
                "A travel notice is only for a cardholder who made the purchase."
            )
        if until < today:
            raise BankError(
                "The return date has already passed. Ask when they're back."
            )
        if (until - today).days > LONGEST_TRAVEL_NOTICE_DAYS:
            raise BankError(
                f"A travel notice can cover at most {LONGEST_TRAVEL_NOTICE_DAYS} days."
            )
        alert.travel = TravelNotice(destination.strip(), until)
        await self._publish_alert()
        return alert

    def _require_alert(self) -> Alert:
        if self.alert is None:
            raise BankError("There's no flagged purchase on this call.")
        return self.alert

    def _require_verified(self) -> Alert:
        alert = self._require_alert()
        if self.state != "authenticated":
            raise BankError("Verify the cardholder's date of birth first.")
        return alert

    async def _publish_alert(self) -> None:
        await self.publish("alert", self._require_alert().to_json())

    async def _set_state(self, state: AuthState) -> None:
        self.state = state
        await self.publish("status", state)


def _canonical(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _cents(amount: str) -> Decimal | None:
    try:
        return Decimal(re.sub(r"[$,\s]", "", amount)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
