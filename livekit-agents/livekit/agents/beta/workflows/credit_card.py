from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from .name import GetNameTask
from .task_group import TaskGroup

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode

CardIssuersLookup = {"3": "American Express", "4": "Visa", "5": "Mastercard", "6": "Discover"}


@dataclass
class GetCreditCardResult:
    cardholder_name: str
    issuer: str
    card_number: int
    security_code: str
    expiration_date: str


@dataclass
class GetCardNumberResult:
    issuer: str
    card_number: int


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


@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
async def decline_card_capture(context: RunContext, reason: str) -> None:
    """Handles the case when the user explicitly declines to provide a detail for their card information.

    Args:
        reason (str): A short explanation of why the user declined to provide card information
    """
    task = context.session.current_agent
    if isinstance(task, AgentTask) and not task.done():
        task.complete(CardCaptureDeclinedError(reason))


@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
async def restart_card_collection(context: RunContext, reason: str) -> None:
    """Handles the case when the user wishes to start over the card information collection process and validate a new card.

    Args:
        reason (str): A short explanation of why the user wishes to start over
    """
    task = context.session.current_agent
    if isinstance(task, AgentTask) and not task.done():
        task.complete(CardCollectionRestartError(reason))


class GetCardNumberTask(AgentTask[GetCardNumberResult]):
    def __init__(
        self,
        *,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(
                "You are a single step in a broader process of collecting credit card information.\n"
                "You are solely responsible for collecting the card number.\n"
                "If the user refuses to provide a number, call decline_card_capture().\n"
                "If the user wishes to start over the card collection process, call restart_card_collection().\n"
                "Avoid listing out questions with bullet points or numbers, use a natural conversational tone.\n"
                "Never repeat any sensitive information, such as the user's card number, back to the user.\n"
            ),
            tools=[decline_card_capture, restart_card_collection],
        )
        self._card_number = 0
        self._require_confirmation = require_confirmation

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Ask for the user's credit card number.",
        )

    @function_tool()
    async def update_card_number(
        self,
        context: RunContext,
        card_number: int,
    ) -> str | None:
        """Call to update the user's card number.

        Args:
            card_number (int): The credit card number
        """
        if len(str(card_number)) < 13 or len(str(card_number)) > 19:
            self.session.generate_reply(
                instructions="The length of the card number is invalid, ask the user to repeat their card number."
            )
            return
        else:
            self._card_number = card_number

            if not self._confirmation_required(context):
                if not self.validate_card_number(self._card_number):
                    self.session.generate_reply(
                        instructions="The card number is not valid, ask the user if they made a mistake or to provide another card."
                    )
                else:
                    first_digit = str(self._card_number)[0]
                    issuer = CardIssuersLookup.get(first_digit, "Other")
                    if not self.done():
                        self.complete(
                            GetCardNumberResult(issuer=issuer, card_number=self._card_number)
                        )
                return None

            confirm_tool = self._build_confirm_tool(card_number=card_number)
            current_tools = [t for t in self.tools if t.id != "confirm_card_number"]
            current_tools.append(confirm_tool)
            await self.update_tools(current_tools)

            return (
                f"The card number has been updated to {card_number}\n"
                f"Ask them to repeat the number, do not repeat the number back to them.\n"
            )

    def _build_confirm_tool(self, *, card_number: int):
        @function_tool()
        async def confirm_card_number(repeated_card_number: int) -> None:
            """Call after the user repeats their card number for confirmation.

            Args:
                repeated_card_number (int): The card number repeated by the user
            """
            if repeated_card_number != card_number:
                self.session.generate_reply(
                    instructions="The repeated card number does not match, ask the user to try again."
                )
                return

            if not self.validate_card_number(card_number):
                self.session.generate_reply(
                    instructions="The card number is not valid, ask the user if they made a mistake or to provide another card."
                )
            else:
                first_digit = str(card_number)[0]
                issuer = CardIssuersLookup.get(first_digit, "Other")
                if not self.done():
                    self.complete(GetCardNumberResult(issuer=issuer, card_number=card_number))

        return confirm_card_number

    def validate_card_number(self, card_number) -> bool:
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

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"


class GetSecurityCodeTask(AgentTask[GetSecurityCodeResult]):
    def __init__(
        self,
        *,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(
                "You are a single step in a broader process of collecting credit card information.\n"
                "You are solely responsible for collecting the user's card's security code.\n"
                "If the user refuses to provide a code, call decline_card_capture().\n"
                "If the user wishes to start over the card collection process, call restart_card_collection().\n"
                "Avoid listing out questions with bullet points or numbers, use a natural conversational tone.\n"
                "Never repeat any sensitive information, such as the user's security code, back to the user.\n"
            ),
            tools=[decline_card_capture, restart_card_collection],
        )
        self._security_code = ""
        self._require_confirmation = require_confirmation

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Collect the user's card's security code.",
        )

    @function_tool()
    async def update_security_code(
        self,
        context: RunContext,
        security_code: str,
    ) -> str | None:
        """Call to update the card's security code.

        Args:
            security_code (str): The card's security code (3-4 digits, may have leading zeros).
        """
        stripped = security_code.strip()
        if not stripped.isdigit() or not (3 <= len(stripped) <= 4):
            self.session.generate_reply(
                instructions="The security code's length is invalid, ask the user to repeat or to provide a new card and start over."
            )
            return
        else:
            self._security_code = stripped

            if not self._confirmation_required(context):
                if not self.done():
                    self.complete(GetSecurityCodeResult(security_code=self._security_code))
                return None

            confirm_tool = self._build_confirm_tool(security_code=stripped)
            current_tools = [t for t in self.tools if t.id != "confirm_security_code"]
            current_tools.append(confirm_tool)
            await self.update_tools(current_tools)

            return (
                f"The security code has been updated to {stripped}\n"
                f"Do not repeat the security code back to the user, ask them to repeat themselves.\n"
            )

    def _build_confirm_tool(self, *, security_code: str) -> llm.FunctionTool:
        @function_tool()
        async def confirm_security_code(repeated_security_code: str) -> None:
            """Call after the user repeats their security code for confirmation.

            Args:
                repeated_security_code (str): The security code repeated by the user
            """
            if repeated_security_code.strip() != security_code:
                self.session.generate_reply(
                    instructions="The repeated security code does not match, ask the user to try again."
                )
                return

            if not self.done():
                self.complete(GetSecurityCodeResult(security_code=security_code))

        return confirm_security_code

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"


class GetExpirationDateTask(AgentTask[GetExpirationDateResult]):
    def __init__(
        self,
        *,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(
                "You are a single step in a broader process of collecting credit card information.\n"
                "You are solely responsible for collecting the user's card's expiration date.\n"
                "If the user refuses to provide a date, call decline_card_capture().\n"
                "If the user wishes to start over the card collection process, call restart_card_collection().\n"
                "Avoid listing out questions with bullet points or numbers, use a natural conversational tone.\n"
                "Never repeat any sensitive information, such as the user's expiration date, back to the user.\n"
            ),
            tools=[decline_card_capture, restart_card_collection],
        )
        self._expiration_date = ""
        self._require_confirmation = require_confirmation

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
    ) -> str | None:
        """Call to update the card's expiration date. Collect both the numerical month and year.

        Args:
            expiration_month (int): The numerical expiration month of the card, example: '04' for April
            expiration_year (int): The numerical expiration year of the card shortened to the last two digits, for example, '35' for 2035
        """
        if not (1 <= expiration_month <= 12):
            self.session.generate_reply(
                instructions="The expiration month is invalid, ask the user to repeat the expiration month."
            )
            return
        elif not (0 <= expiration_year <= 99):
            self.session.generate_reply(
                instructions="The expiration year is invalid, ask the user to repeat the expiration year."
            )
            return
        elif self._is_expired(expiration_month, expiration_year):
            self.session.generate_reply(
                instructions="The expiration date is in the past, the card is expired. Ask the user to provide another card."
            )
            return
        else:
            self._expiration_date = f"{expiration_month:02d}/{expiration_year:02d}"

            if not self._confirmation_required(context):
                if not self.done():
                    self.complete(GetExpirationDateResult(date=self._expiration_date))
                return None

            confirm_tool = self._build_confirm_tool(
                expiration_month=expiration_month, expiration_year=expiration_year
            )
            current_tools = [t for t in self.tools if t.id != "confirm_expiration_date"]
            current_tools.append(confirm_tool)
            await self.update_tools(current_tools)

            return (
                f"The expiration date has been updated to {self._expiration_date}\n"
                f"Do not repeat the expiration date back to the user, ask them to repeat themselves.\n"
                # f"Do not call `confirm_expiration_date` directly"
            )

    def _build_confirm_tool(
        self, *, expiration_month: int, expiration_year: int
    ) -> llm.FunctionTool:
        expiration_date = self._expiration_date

        @function_tool()
        async def confirm_expiration_date(
            repeated_expiration_month: int,
            repeated_expiration_year: int,
        ) -> None:
            """Call after the user repeats their expiration date for confirmation.

            Args:
                repeated_expiration_month (int): The expiration month repeated by the user
                repeated_expiration_year (int): The expiration year repeated by the user
            """
            if (
                repeated_expiration_month != expiration_month
                or repeated_expiration_year != expiration_year
            ):
                self.session.generate_reply(
                    instructions="The repeated expiration date does not match, ask the user to try again."
                )
                return

            if not self.done():
                self.complete(GetExpirationDateResult(date=expiration_date))

        return confirm_expiration_date

    def _is_expired(self, month: int, year: int) -> bool:
        today = date.today()
        full_year = 2000 + year
        return (full_year, month) < (today.year, today.month)

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"


class GetCreditCardTask(AgentTask[GetCreditCardResult]):
    def __init__(
        self,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
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
        self._require_confirmation = require_confirmation

    async def on_enter(self) -> None:
        while not self.done():
            task_group = TaskGroup()
            task_group.add(
                lambda: GetNameTask(
                    last_name=True,
                    extra_instructions="This is in the context of credit card information collection, ask specifically for the full name listed on it.",
                    require_confirmation=self._require_confirmation,
                ),
                id="cardholder_name_task",
                description="Collects the cardholder's full name",
            )
            task_group.add(
                lambda: GetCardNumberTask(require_confirmation=self._require_confirmation),
                id="card_number_task",
                description="Collects the user's card number",
            )
            task_group.add(
                lambda: GetSecurityCodeTask(require_confirmation=self._require_confirmation),
                id="security_code_task",
                description="Collects the card's security code",
            )
            task_group.add(
                lambda: GetExpirationDateTask(require_confirmation=self._require_confirmation),
                id="expiration_date_task",
                description="Collects the card's expiration date",
            )
            try:
                results = await task_group
                name = f"{results.task_results['cardholder_name_task'].first_name} {results.task_results['cardholder_name_task'].last_name}"
                result = GetCreditCardResult(
                    cardholder_name=name,
                    issuer=results.task_results["card_number_task"].issuer,
                    card_number=results.task_results["card_number_task"].card_number,
                    security_code=results.task_results["security_code_task"].security_code,
                    expiration_date=results.task_results["expiration_date_task"].date,
                )
                self.complete(result)
            except CardCollectionRestartError:
                continue
            except (CardCaptureDeclinedError, ToolError) as e:
                self.complete(e)
