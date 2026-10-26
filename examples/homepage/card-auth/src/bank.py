from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

BANK_NAME = "Kestrel Bank"

# Nothing in the demo is on file. The card number alone decides the identity check: a card ending in
# VERIFIED_CARD always passes and any other card fails, like DECLINED_CARD, the one the demo offers
# for seeing a failure.
VERIFIED_CARD = "1111"
DECLINED_CARD = "0000"
MAX_ATTEMPTS = 3
EARLIEST_BIRTH_DATE = date(1900, 1, 1)

AuthState = Literal["unauthenticated", "authenticating", "authenticated"]
CardState = Literal["active", "locked"]


class BankError(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    name: str
    date_of_birth: date
    last_four: str

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name,
            "dateOfBirth": self.date_of_birth.isoformat(),
            "lastFour": self.last_four,
        }


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
class Card:
    """The demo card, who it was verified as, and what the call changed on it."""

    last_four: str = VERIFIED_CARD
    holder: str | None = None
    state: CardState = "active"
    case_id: str | None = None
    travel: TravelNotice | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "lastFour": self.last_four,
            "holder": self.holder,
            "state": self.state,
            "caseId": self.case_id,
            "travel": None
            if self.travel is None
            else {
                "destination": self.travel.destination,
                "until": self.travel.until.isoformat(),
            },
        }


@dataclass
class Alert:
    """A flagged purchase on the card: the one the outbound call is about, or the charge an inbound
    caller doesn't recognize. `checks` are the outbound call's identity checks."""

    purchase: FlaggedPurchase = FLAGGED
    checks: list[bool | None] = field(default_factory=list)
    made_by_cardholder: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "purchase": self.purchase.to_json(),
            "checks": self.checks,
            "madeByCardholder": self.made_by_cardholder,
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
    locked_out: bool = False
    card: Card = field(default_factory=Card)
    alert: Alert | None = None

    async def authenticate(self, claimed: Identity) -> bool:
        """Inbound: checks the caller against the demo's test cards. Three misses lock them out."""
        if self.locked_out:
            raise BankError("This caller is locked out after too many failed checks.")
        attempt: dict[str, Any] = {
            "submitted": claimed.to_json(),
            "authenticated": None,
        }
        self.attempts.append(attempt)
        await self.publish("attempts", self.attempts)
        await self._set_state("authenticating")
        await asyncio.sleep(self.lookup_seconds)
        verified = claimed.last_four == VERIFIED_CARD
        attempt["authenticated"] = verified
        await self.publish("attempts", self.attempts)
        if verified:
            await self._verify_holder(claimed.name)
        elif sum(a["authenticated"] is False for a in self.attempts) >= MAX_ATTEMPTS:
            self.locked_out = True
            await self.publish("lockedOut", True)
        await self._set_state("authenticated" if verified else "unauthenticated")
        return verified

    async def open_alert(self) -> Alert:
        self.alert = Alert()
        await self._publish_alert()
        await self._publish_card()
        return self.alert

    async def confirm_cardholder(self, name: str, date_of_birth: date) -> None:
        """Outbound: confirms whoever picked up as the cardholder. Nothing is on file in the demo, so
        any name and real date of birth will do; the check is there to show the step."""
        alert = self._require_alert()
        alert.checks.append(None)
        await self._publish_alert()
        await self._set_state("authenticating")
        await asyncio.sleep(self.lookup_seconds)
        alert.checks[-1] = True
        await self._publish_alert()
        await self._verify_holder(name)
        await self._set_state("authenticated")

    async def recent_charges(self) -> Alert:
        """Inbound: the charge a verified caller is asking about, the bank's only recent one."""
        self._require_verified()
        if self.alert is None:
            self.alert = Alert()
            await self._publish_alert()
        return self.alert

    async def record_answer(self, made_by_cardholder: bool) -> Card:
        self._require_verified()
        alert = self._require_alert()
        alert.made_by_cardholder = made_by_cardholder
        if made_by_cardholder:
            if self.card.case_id is not None:
                self.card.state = "active"
                self.card.case_id = None
        else:
            self.card.state = "locked"
            self.card.case_id = LOSS_PREVENTION_CASE
            self.card.travel = None
        await self._publish_alert()
        await self._publish_card()
        return self.card

    async def lock_card(self) -> Card:
        self._require_verified()
        self.card.state = "locked"
        self.card.travel = None
        await self._publish_card()
        return self.card

    async def set_travel_notice(
        self, destination: str, until: date, today: date
    ) -> Card:
        self._require_verified()
        if self.card.state == "locked":
            raise BankError(
                "The card is locked, so a travel notice wouldn't do anything."
            )
        if until < today:
            raise BankError(
                "The return date has already passed. Ask when they're back."
            )
        if (until - today).days > LONGEST_TRAVEL_NOTICE_DAYS:
            raise BankError(
                f"A travel notice can cover at most {LONGEST_TRAVEL_NOTICE_DAYS} days."
            )
        self.card.travel = TravelNotice(destination.strip(), until)
        await self._publish_card()
        return self.card

    def _require_alert(self) -> Alert:
        if self.alert is None:
            raise BankError("There's no flagged purchase on this call.")
        return self.alert

    def _require_verified(self) -> None:
        if self.state != "authenticated":
            raise BankError("Verify the caller first.")

    async def _verify_holder(self, name: str) -> None:
        self.card.holder = name
        await self._publish_card()

    async def _publish_alert(self) -> None:
        await self.publish("alert", self._require_alert().to_json())

    async def _publish_card(self) -> None:
        await self.publish("card", self.card.to_json())

    async def _set_state(self, state: AuthState) -> None:
        self.state = state
        await self.publish("status", state)


def card_digits(text: str) -> str:
    """The card's last four digits as the backend model gave them, like "1111" or "11 11"."""
    digits = re.sub(r"\D", "", text)
    if len(digits) != 4:
        raise BankError(f"Ask for the card's last four digits, got {text!r}.")
    return digits


def birth_date(text: str, today: date) -> date:
    """A date of birth as the backend model gave it, YYYY-MM-DD, if it could be someone's."""
    try:
        born = date.fromisoformat(text.strip())
    except ValueError as e:
        raise BankError(f"Use YYYY-MM-DD for the date of birth, got {text!r}.") from e
    if not EARLIEST_BIRTH_DATE <= born < today:
        raise BankError("That can't be a date of birth. Ask for it again.")
    return born
