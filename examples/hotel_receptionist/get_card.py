from __future__ import annotations

from dataclasses import dataclass

from hotel_db import TODAY
from persona import COMMON_INSTRUCTIONS

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_ISSUERS = {"3": "American Express", "4": "Visa", "5": "Mastercard", "6": "Discover"}

_CARD_INSTRUCTIONS = """\
You're collecting the caller's credit card - the whole card in this one step: number, expiration date, security code, and the name on the card.

Take details in whatever order the caller offers them, recording each with its tool the moment you have it. The natural asking order: number, expiration, security code, then the name on the card - the name is often already in the conversation, so confirm it rather than re-asking. Each tool's return names the next step; follow it.

Expect noisy voice transcription: digits read aloud ('four' -> 4, 'oh'/'zero' -> 0), expiration dates like 'oh four twenty five', 'four slash twenty five', or 'April twenty twenty-five'. Normalize silently and filter filler words. Only record the card number once the caller has given the entire number - never in increments.

Never read the full card number or the security code back to the caller; refer to the card by its last four digits only. If a tool rejects a value, ask the caller to repeat just that detail - don't start the whole card over. If the caller switches cards mid-way, just record the new values; recording a field again replaces it.

If the caller refuses to provide the card, call decline_card_capture.
"""


@dataclass
class GetCardResult:
    cardholder_name: str
    issuer: str
    card_number: str
    security_code: str
    expiration_date: str


def _luhn_ok(card_number: str) -> bool:
    total = 0
    for index, digit in enumerate(card_number[::-1]):
        n = int(digit)
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class GetCardTask(AgentTask[GetCardResult]):
    """The whole card capture as ONE task: four recording tools on a single
    agent instead of a sub-task per field. Validation lives in ToolErrors
    (Luhn, expiry, code length) so a bad value bounces straight back to the
    model with instructions to re-ask just that field; one verbal read-back
    (last four + expiry) gates confirm_card()."""

    def __init__(self, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._card_number: str = ""
        self._expiration: str = ""
        self._security_code: str = ""
        self._first_name: str = ""
        self._last_name: str = ""
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_CARD_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Take the caller's card details. Scan the conversation first - if any "
                "card detail was already given, record it rather than re-asking - then "
                "ask for the card number."
            )
        )

    def _status(self) -> str:
        # Next-action directive, not a missing-field list (field names leak
        # into the spoken question otherwise).
        if not self._card_number:
            return "next: ask for the card number, then call record_card_number"
        if not self._expiration:
            return "next: ask for the expiration date, then call record_expiration"
        if not self._security_code:
            return "next: ask for the security code, then call record_security_code"
        if not (self._first_name and self._last_name):
            return "next: confirm the name on the card, then call record_cardholder"
        return (
            "all card details captured - read the last four digits and expiration back "
            "to the caller, and once they agree, call confirm_card()"
        )

    @function_tool()
    async def record_card_number(self, card_number: str) -> str:
        """Record the card number, only once the caller has given the entire number.

        Args:
            card_number: All the digits, no spaces or dashes.
        """
        digits = "".join(c for c in card_number if c.isdigit())
        if not 13 <= len(digits) <= 19:
            raise ToolError(
                "that card number has the wrong number of digits - ask the caller to read it again"
            )
        if not _luhn_ok(digits):
            raise ToolError(
                "that number fails the card check, one digit is likely off - "
                "ask the caller to read it again slowly"
            )
        self._card_number = digits
        return f"card number recorded (ending {digits[-4:]}) | {self._status()}"

    @function_tool()
    async def record_expiration(self, month: int, year: int) -> str:
        """Record the card's expiration date.

        Args:
            month: Expiration month as a number, e.g. 4 for April.
            year: Expiration year, last two digits, e.g. 28 for 2028.
        """
        if not 1 <= month <= 12:
            raise ToolError("that expiration month is invalid - ask the caller to repeat it")
        if not 0 <= year <= 99:
            raise ToolError("that expiration year is invalid - ask the caller to repeat it")
        if (2000 + year, month) < (TODAY.year, TODAY.month):
            raise ToolError("that date is in the past, the card is expired - ask for another card")
        self._expiration = f"{month:02d}/{year:02d}"
        return f"expiration recorded | {self._status()}"

    @function_tool()
    async def record_security_code(self, security_code: str) -> str:
        """Record the card's security code.

        Args:
            security_code: The 3 or 4 digit code, leading zeros included.
        """
        code = security_code.strip()
        if not code.isdigit() or not 3 <= len(code) <= 4:
            raise ToolError(
                "the security code should be 3 or 4 digits - ask the caller to repeat it"
            )
        self._security_code = code
        return f"security code recorded | {self._status()}"

    @function_tool()
    async def record_cardholder(self, first_name: str, last_name: str) -> str:
        """Record the name as it appears on the card.

        Args:
            first_name: Cardholder's first name, exactly as given.
            last_name: Cardholder's last name, exactly as given.
        """
        first_name, last_name = first_name.strip(), last_name.strip()
        for label, value in (("first", first_name), ("last", last_name)):
            if not value or not any(c.isalpha() for c in value):
                raise ToolError(f"{label} name {value!r} doesn't look like a name - ask again")
        self._first_name, self._last_name = first_name, last_name
        return f"cardholder recorded: {first_name} {last_name} | {self._status()}"

    @function_tool()
    async def confirm_card(self) -> None:
        """Finalize the card capture. Call only after the caller has agreed to the read-back of the last four digits and expiration."""
        if not (
            self._card_number
            and self._expiration
            and self._security_code
            and self._first_name
            and self._last_name
        ):
            raise ToolError(self._status())
        if not self.done():
            self.complete(
                GetCardResult(
                    cardholder_name=f"{self._first_name} {self._last_name}",
                    issuer=_ISSUERS.get(self._card_number[0], "Other"),
                    card_number=self._card_number,
                    security_code=self._security_code,
                    expiration_date=self._expiration,
                )
            )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_card_capture(self, reason: str) -> None:
        """The caller explicitly refuses to provide their card.

        Args:
            reason: A short explanation of why the caller declined.
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the card details: {reason}"))
