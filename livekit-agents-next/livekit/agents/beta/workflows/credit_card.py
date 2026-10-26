from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ...llm.tool_context import ToolError

CARD_ISSUERS_LOOKUP = {"3": "American Express", "4": "Visa", "5": "Mastercard", "6": "Discover"}


@dataclass
class GetCreditCardResult:
    cardholder_name: str
    issuer: str
    card_number: str
    security_code: str
    expiration_date: str


@dataclass
class GetCardNumberResult:
    issuer: str
    card_number: str


@dataclass
class GetSecurityCodeResult:
    security_code: str


@dataclass
class GetExpirationDateResult:
    date: str


class CardCaptureDeclinedError(ToolError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"couldn't get the card details: {reason}")
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason


class CardCollectionRestartError(ToolError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"starting over: {reason}")
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason


def validate_card_number(card_number: str) -> bool:
    """Validates card number via the Luhn algorithm"""
    if not card_number or not card_number.isdigit():
        return False
    total_sum = 0

    reversed_number = card_number[::-1]
    for index, digit in enumerate(reversed_number):
        if index % 2 == 1:
            doubled_digit = int(digit) * 2
            if doubled_digit > 9:
                total_sum += doubled_digit - 9
            else:
                total_sum += doubled_digit
        else:
            total_sum += int(digit)

    return total_sum % 10 == 0


def is_expired(month: int, year: int) -> bool:
    """Whether a two-digit card expiration year/month precedes this month."""
    today = date.today()
    return (2000 + year, month) < (today.year, today.month)
