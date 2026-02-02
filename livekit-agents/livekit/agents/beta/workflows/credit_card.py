from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from ...voice.speech_handle import SpeechHandle
from . import TaskGroup

if TYPE_CHECKING:
    from ...voice.agent_session import TurnDetectionMode

CardIssuersLookup = {"3": "American Express", "4": "Visa", "5": "Mastercard", "6": "Discover"}


@dataclass
class GetCreditCardResult:
    cardholder_name: str
    issuer: str
    card_number: int
    security_code: int
    expiration_date: str


@dataclass
class GetCardHolderNameResult:
    full_name: str


@dataclass
class GetCardNumberResult:
    issuer: str
    card_number: int


@dataclass
class GetSecurityCodeResult:
    security_code: int


@dataclass
class GetExpirationDateResult:
    date: str


@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
async def decline_card_capture(self, reason: str) -> None:
    """Handles the case when the user explicitly declines to provide a detail for their card information.

    Args:
        reason (str): A short explanation of why the user declined to provide card information
    """
    if not self.done():
        self.complete(ToolError(f"couldn't get the card details: {reason}"))


@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
async def restart_card_collection(self, reason: str) -> None:
    """Handles the case when the user wishes to start over the card information collection process and validate a new card.

    Args:
        reason (str): A short explanation of why the user wishes to start over
    """
    if not self.done():
        self.complete(ToolError(f"starting over: {reason}"))


class GetCardHolderNameTask(AgentTask[GetCardHolderNameResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a single step in a broader process of collecting credit card information.
            You are solely responsible for collecting the cardholder's full name.
            If the user refuses to provide a name, call decline_card_capture().
            Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
            """,
            tools=[decline_card_capture],
        )
        self._full_name = ""
        self._name_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Collect the cardholder's full name.",
        )

    @function_tool()
    async def update_cardholder_name(
        self,
        context: RunContext,
        full_name: str,
    ) -> None:
        """Updates the cardholder's full name.

        Args:
            full_name (str): The cardholder's full name
        """
        self._name_update_speech_handle = context.speech_handle

        self._full_name = full_name
        return (
            f"The cardholder's name has been updated to {full_name}\n"
            f"Repeat the name character by character: {full_name} if needed\n"
            f"Prompt the user for confirmation, do not call `confirm_cardholder_name` directly"
        )

    @function_tool()
    async def confirm_cardholder_name(
        self,
        context: RunContext,
    ) -> None:
        """Confirms the cardholder's full name."""
        await context.wait_for_playout()

        if context.speech_handle == self._name_update_speech_handle:
            raise ToolError("error: the user must confirm the cardholder name explicitly")
        if not self._full_name:
            raise ToolError(
                "error: no name was provided, 'update_cardholder_name must be called prior"
            )
        if not self.done():
            self.complete(GetCardHolderNameResult(full_name=self._full_name))


class GetCardNumberTask(AgentTask[GetCardNumberResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a single step in a broader process of collecting credit card information.
            You are solely responsible for collecting the card number.
            If the user refuses to provide a number, call decline_card_capture().
            If the user wishes to start over the card collection process, call restart_card_collection().
            Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
            Be sure to confirm the card number by reading it out digit-by-digit.
            """,
            tools=[decline_card_capture, restart_card_collection],
        )
        self._card_number = 0
        self._number_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Collect the user's credit card number.",
        )

    @function_tool()
    async def update_card_number(
        self,
        context: RunContext,
        card_number: int,
    ) -> None:
        """Call to update the user's card number.

        Args:
            card_number (int): The credit card number
        """
        if len(str(card_number)) < 13 and len(str(card_number)) > 19:
            self.session.generate_reply(
                instructions="The length of the card number is invalid, ask the user to repeat their card number."
            )
        else:
            self._number_update_speech_handle = context.speech_handle
            self._card_number = card_number
            return (
                f"The card number has been updated to {card_number}\n"
                f"Repeat the card number digit-by-digit: {card_number}\n"
                f"Prompt the user for confirmation, do not call `confirm_card_number` directly"
            )

    @function_tool()
    async def confirm_card_number(
        self,
        context: RunContext,
    ) -> None:
        """Call after confirming the user's card number."""
        await context.wait_for_playout()

        if context.speech_handle == self._number_update_speech_handle:
            raise ToolError("error: the user must confirm the card number explicitly")

        if not self._card_number:
            raise ToolError(
                "error: no card number was provided, `update_card_number` must be called prior"
            )

        if not self.validate_card_number(self._card_number):
            self.session.generate_reply(
                instructions="The card number is not valid, ask the user if they made a mistake or to provide another card."
            )
        else:
            first_digit = str(self._card_number)[0]
            issuer = CardIssuersLookup.get(first_digit, "Other")
            self.complete(GetCardNumberResult(issuer=issuer, card_number=self._card_number))

    def validate_card_number(self, card_number):
        """Validates card number via the Luhn algorithm"""
        total_sum = 0

        reversed_number = str(card_number)[::-1]
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


class GetSecurityCodeTask(AgentTask[GetSecurityCodeResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a single step in a broader process of collecting credit card information.
            You are solely responsible for collecting the user's card's security code.
            If the user refuses to provide a code, call decline_card_capture().
            If the user wishes to start over the card collection process, call restart_card_collection().
            Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
            Be sure to confirm the security code by reading it out digit-by-digit.
            """,
            tools=[decline_card_capture, restart_card_collection],
        )
        self._security_code = 0
        self._security_code_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Collect the user's card's security code.",
        )

    @function_tool()
    async def update_security_code(
        self,
        context: RunContext,
        security_code: int,
    ) -> None:
        """Call to update the card's security code.

        Args:
            security_code (int): The card's security code.
        """
        if len(str(security_code)) < 3 and len(str(security_code)) > 4:
            self.session.generate_reply(
                instructions="The security code's length is invalid, ask the user to repeat or to provide a new card and start over."
            )
        else:
            self._security_code_update_speech_handle = context.speech_handle
            self._security_code = security_code
            return (
                f"The security code has been updated to {security_code}\n"
                f"Repeat the security code digit-by-digit: {security_code}\n"
                f"Prompt the user for confirmation, do not call `confirm_security_code` directly"
            )

    @function_tool()
    async def confirm_security_code(
        self,
        context: RunContext,
    ) -> None:
        """Call after confirming the card's security code."""
        await context.wait_for_playout()

        if context.speech_handle == self._security_code_update_speech_handle:
            raise ToolError("error: the user must confirm the security code explicitly")

        if not self._security_code:
            raise ToolError(
                "error: no security code was provided, `update_security_code` must be called prior"
            )

        self.complete(GetSecurityCodeResult(security_code=self._security_code))


class GetExpirationDateTask(AgentTask[GetExpirationDateResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a single step in a broader process of collecting credit card information.
            You are solely responsible for collecting the user's card's expiration date.
            If the user refuses to provide a date, call decline_card_capture().
            If the user wishes to start over the card collection process, call restart_card_collection().
            Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
            Be sure to confirm the expiration date by repeating the month and year back to the user.
            """,
            tools=[decline_card_capture, restart_card_collection],
        )
        self._expiration_date = ""
        self._date_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Collect the user's card's expiration date.",
        )

    @function_tool()
    async def update_expiration_date(
        self,
        context: RunContext,
        expiration_month: int,
        expiration_year: int,
    ) -> None:
        """Call to update the card's expiration date. Collect both the numerical month and year.

        Args:
            expiration_month (int): The numerical expiration month of the card, example: '04' for April
            expiration_year (int): The numerical expiration year of the card shortened to the last two digits, for example, '35' for 2035
        """
        if len(str(expiration_month)) != 2:
            self.session.generate_reply(
                instructions="The expiration month has not been formatted correctly, ask the user to repeat the expiration month."
            )
        if len(str(expiration_year)) != 2:
            self.session.generate_reply(
                instructions="The expiration year has not been formatted correctly, ask the user to repeat the expiration year."
            )
        else:
            self._date_update_speech_handle = context.speech_handle
            self._expiration_date = expiration_month + "/" + expiration_year
            return (
                f"The expiration date has been updated to {self._expiration_date}\n"
                f"Repeat the expiration date by stating the corresponding month and year: {self._expiration_date}\n"
                f"Prompt the user for confirmation, do not call `confirm_expiration_date` directly"
            )

    @function_tool()
    async def confirm_expiration_date(
        self,
        context: RunContext,
    ) -> None:
        """Call after confirming the user's card's expiration date."""
        await context.wait_for_playout()
        if context.speech_handle == self._date_update_speech_handle:
            raise ToolError("error: the user must confirm the expiration date explicitly")

        if not self._expiration_date:
            raise ToolError(
                "error: no expiration date was provided, `update_expiration_date` must be called before"
            )
        self.complete(GetExpirationDateResult(date=self._expiration_date))


class GetCreditCardTask(AgentTask[GetCreditCardResult]):
    def __init__(
        self,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions="*none*",
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

    async def on_enter(self) -> None:
        while not self.done():
            task_group = TaskGroup()
            task_group.add(
                lambda: GetCardHolderNameTask(),
                id="cardholder_name_task",
                description="Collects the cardholder's full name",
            )
            task_group.add(
                lambda: GetCardNumberTask(),
                id="card_number_task",
                description="Collects the user's card number",
            )
            task_group.add(
                lambda: GetSecurityCodeTask(),
                id="security_code_task",
                description="Collects the card's security code",
            )
            task_group.add(
                lambda: GetExpirationDateTask(),
                id="get_expiration_date_task",
                description="Collects the card's expiration date",
            )
            try:
                results = await task_group
                result = GetCreditCardResult(
                    cardholder_name=results["cardholder_name_task"].full_name,
                    issuer=results["card_number_task"].issuer,
                    card_number=results["card_number_task"].card_number,
                    security_code=results["security_code_task"].security_code,
                    expiration_date=results["get_expiration_date"].date,
                )
                self.complete(result)
            except ToolError as e:
                if "starting over" in e.message:
                    continue
                else:
                    self.complete(e)
