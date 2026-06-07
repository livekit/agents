from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.chat_context import Instructions
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from .name import GetNameTask
from .task_group import TaskGroup

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode

CARD_ISSUERS_LOOKUP = {"3": "American Express", "4": "Visa", "5": "Mastercard", "6": "Discover"}

_CARD_NUMBER_BASE_INSTRUCTIONS = """
You are a single step in a broader process of collecting credit card information.
You are solely responsible for collecting the credit card number.
{modality_specific}
If the user refuses to provide a credit card number, call decline_card_capture().
If the user wishes to start over the credit card collection process, call restart_card_collection().
Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
Never repeat any sensitive information, such as the user's credit card number, back to the user.
{confirmation_instructions}
"""

_CARD_NUMBER_AUDIO_SPECIFIC = """
Handle input as noisy voice transcription. Expect users to read the card number digit by digit.
Normalize spoken digits silently: 'four' → 4, 'zero' / 'oh' → 0.
Filter out filler words or hesitations.
"""

_CARD_NUMBER_TEXT_SPECIFIC = """
Handle input as typed text. Users may type the number with or without spaces or dashes (e.g. '4152 6374 8901 2345').
"""

_SECURITY_CODE_BASE_INSTRUCTIONS = """
You are a single step in a broader process of collecting credit card information.
You are solely responsible for collecting the user's card's security code.
{modality_specific}
If the user refuses to provide a code, call decline_card_capture().
If the user wishes to start over the card collection process, call restart_card_collection().
Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
Never repeat any sensitive information, such as the user's security code, back to the user.
{confirmation_instructions}
"""

_SECURITY_CODE_AUDIO_SPECIFIC = """
Handle input as noisy voice transcription. Expect users to read the security code digit by digit.
Normalize spoken digits silently: 'four' → 4, 'zero' / 'oh' → 0.
Filter out filler words or hesitations.
"""

_SECURITY_CODE_TEXT_SPECIFIC = """
Handle input as typed text. Users will type the security code directly.
"""

_EXPIRATION_DATE_BASE_INSTRUCTIONS = """
You are a single step in a broader process of collecting credit card information.
You are solely responsible for collecting the user's card's expiration date.
{modality_specific}
If the user refuses to provide a date, call decline_card_capture().
If the user wishes to start over the card collection process, call restart_card_collection().
Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
Never repeat any sensitive information, such as the user's expiration date, back to the user.
{confirmation_instructions}
"""

_EXPIRATION_DATE_AUDIO_SPECIFIC = """
Handle input as noisy voice transcription. Expect users to say the expiration date in formats like 'April twenty five', 'oh four twenty five', 'four slash twenty five', or 'April 2025'.
Normalize spoken months and digits silently.
Filter out filler words or hesitations.
"""

_EXPIRATION_DATE_TEXT_SPECIFIC = """
Handle input as typed text. Expect users to type the expiration date in formats like '04/25', '04/2025', or 'April 2025'.
"""


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
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
        require_explicit_ask: bool = False,
        extra_instructions: str = "",
    ) -> None:
        confirmation_instructions = (
            "Call `confirm_card_number` once the user has repeated their card number."
        )
        extra_suffix = f"\n{extra_instructions}" if extra_instructions else ""

        self._card_number = ""
        self._require_confirmation = require_confirmation
        self._require_explicit_ask = require_explicit_ask

        super().__init__(
            instructions=Instructions(
                _CARD_NUMBER_BASE_INSTRUCTIONS.format(
                    modality_specific=_CARD_NUMBER_AUDIO_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is not False else ""
                    ),
                )
                + extra_suffix,
                text=_CARD_NUMBER_BASE_INSTRUCTIONS.format(
                    modality_specific=_CARD_NUMBER_TEXT_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is True else ""
                    ),
                )
                + extra_suffix,
            ),
            chat_ctx=chat_ctx,
            tools=[
                self._build_update_card_number_tool(),
                decline_card_capture,
                restart_card_collection,
            ],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Get the user's credit card number. First scan the conversation - if a "
                "credit card number was already given (e.g. the user volunteered it "
                "before the task started), use it via update_card_number rather than "
                "re-asking. Only ask fresh when no credit card number is in the "
                "conversation yet."
            )
        )

    def _build_update_card_number_tool(self) -> llm.FunctionTool:
        # Built dynamically so we can apply IGNORE_ON_ENTER per-instance
        # based on require_explicit_ask.
        flags = ToolFlag.IGNORE_ON_ENTER if self._require_explicit_ask else ToolFlag.NONE

        @function_tool(flags=flags)
        async def update_card_number(context: RunContext, card_number: str) -> str | None:
            """Call to record the user's card number. Only call once the entire number has been given, do not call in increments.

            Args:
                card_number (str): The credit card number as a string with no dashes or spaces
            """
            return await self._update_card_number_impl(context, card_number)

        return update_card_number

    async def _update_card_number_impl(self, context: RunContext, card_number: str) -> str | None:
        card_number = "".join([d for d in card_number if d.isdigit()])
        if len(card_number) < 13 or len(card_number) > 19:
            self.session.generate_reply(
                instructions="The length of the card number is invalid, ask the user to repeat their card number."
            )
            return None
        else:
            self._card_number = card_number

            if not self._confirmation_required(context):
                if not self.validate_card_number(self._card_number):
                    self.session.generate_reply(
                        instructions="The card number is not valid, ask the user if they made a mistake or to provide another card."
                    )
                else:
                    issuer = CARD_ISSUERS_LOOKUP.get(self._card_number[0], "Other")
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
                "The card number has been updated.\n"
                "Ask them to repeat the number, do not repeat the number back to them.\n"
            )

    def _build_confirm_tool(self, *, card_number: str) -> llm.FunctionTool:
        @function_tool()
        async def confirm_card_number(repeated_card_number: str) -> None:
            """Call after the user repeats their card number for confirmation.

            Args:
                repeated_card_number (str): The card number repeated by the user as a string
            """
            repeated_card_number = "".join([d for d in repeated_card_number if d.isdigit()])
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
                issuer = CARD_ISSUERS_LOOKUP.get(card_number[0], "Other")
                if not self.done():
                    self.complete(GetCardNumberResult(issuer=issuer, card_number=card_number))

        return confirm_card_number

    def validate_card_number(self, card_number: str) -> bool:
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

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"


class GetSecurityCodeTask(AgentTask[GetSecurityCodeResult]):
    def __init__(
        self,
        *,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
        require_explicit_ask: bool = False,
        extra_instructions: str = "",
    ) -> None:
        confirmation_instructions = (
            "Call `confirm_security_code` once the user has repeated their security code."
        )
        extra_suffix = f"\n{extra_instructions}" if extra_instructions else ""

        self._security_code = ""
        self._require_confirmation = require_confirmation
        self._require_explicit_ask = require_explicit_ask

        super().__init__(
            instructions=Instructions(
                _SECURITY_CODE_BASE_INSTRUCTIONS.format(
                    modality_specific=_SECURITY_CODE_AUDIO_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is not False else ""
                    ),
                )
                + extra_suffix,
                text=_SECURITY_CODE_BASE_INSTRUCTIONS.format(
                    modality_specific=_SECURITY_CODE_TEXT_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is True else ""
                    ),
                )
                + extra_suffix,
            ),
            chat_ctx=chat_ctx,
            tools=[
                self._build_update_security_code_tool(),
                decline_card_capture,
                restart_card_collection,
            ],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Get the user's card security code. First scan the conversation - if a "
                "code was already given, use it via update_security_code rather than "
                "re-asking. Only ask fresh when no code is in the conversation yet."
            )
        )

    def _build_update_security_code_tool(self) -> llm.FunctionTool:
        # Built dynamically so we can apply IGNORE_ON_ENTER per-instance
        # based on require_explicit_ask.
        flags = ToolFlag.IGNORE_ON_ENTER if self._require_explicit_ask else ToolFlag.NONE

        @function_tool(flags=flags)
        async def update_security_code(context: RunContext, security_code: str) -> str | None:
            """Call to update the card's security code.

            Args:
                security_code (str): The card's security code (3-4 digits, may have leading zeros).
            """
            return await self._update_security_code_impl(context, security_code)

        return update_security_code

    async def _update_security_code_impl(
        self, context: RunContext, security_code: str
    ) -> str | None:
        stripped = security_code.strip()
        if not stripped.isdigit() or not (3 <= len(stripped) <= 4):
            self.session.generate_reply(
                instructions="The security code's length is invalid, ask the user to repeat or to provide a new card and start over."
            )
            return None
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
                "The security code has been updated.\n"
                "Do not repeat the security code back to the user, ask them to repeat themselves.\n"
                "Call `confirm_security_code` once the user confirms, do not call it preemptively.\n"
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
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
        require_explicit_ask: bool = False,
        extra_instructions: str = "",
    ) -> None:
        confirmation_instructions = (
            "Call `confirm_expiration_date` once the user has repeated their expiration date."
        )
        extra_suffix = f"\n{extra_instructions}" if extra_instructions else ""

        self._expiration_date = ""
        self._require_confirmation = require_confirmation
        self._require_explicit_ask = require_explicit_ask

        super().__init__(
            instructions=Instructions(
                _EXPIRATION_DATE_BASE_INSTRUCTIONS.format(
                    modality_specific=_EXPIRATION_DATE_AUDIO_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is not False else ""
                    ),
                )
                + extra_suffix,
                text=_EXPIRATION_DATE_BASE_INSTRUCTIONS.format(
                    modality_specific=_EXPIRATION_DATE_TEXT_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is True else ""
                    ),
                )
                + extra_suffix,
            ),
            chat_ctx=chat_ctx,
            tools=[
                self._build_update_expiration_date_tool(),
                decline_card_capture,
                restart_card_collection,
            ],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Get the user's card expiration date. First scan the conversation - if "
                "an expiration date was already given, use it via update_expiration_date "
                "rather than re-asking. Only ask fresh when no date is in the conversation yet."
            )
        )

    def _build_update_expiration_date_tool(self) -> llm.FunctionTool:
        # Built dynamically so we can apply IGNORE_ON_ENTER per-instance
        # based on require_explicit_ask.
        flags = ToolFlag.IGNORE_ON_ENTER if self._require_explicit_ask else ToolFlag.NONE

        @function_tool(flags=flags)
        async def update_expiration_date(
            context: RunContext, expiration_month: int, expiration_year: int
        ) -> str | None:
            """Call to update the card's expiration date. Collect both the numerical month and year.

            Args:
                expiration_month (int): The numerical expiration month of the card, example: '04' for April
                expiration_year (int): The numerical expiration year of the card shortened to the last two digits, for example, '35' for 2035
            """
            return await self._update_expiration_date_impl(
                context, expiration_month, expiration_year
            )

        return update_expiration_date

    async def _update_expiration_date_impl(
        self, context: RunContext, expiration_month: int, expiration_year: int
    ) -> str | None:
        if not (1 <= expiration_month <= 12):
            self.session.generate_reply(
                instructions="The expiration month is invalid, ask the user to repeat the expiration month."
            )
            return None
        elif not (0 <= expiration_year <= 99):
            self.session.generate_reply(
                instructions="The expiration year is invalid, ask the user to repeat the expiration year."
            )
            return None
        elif self._is_expired(expiration_month, expiration_year):
            self.session.generate_reply(
                instructions="The expiration date is in the past, the card is expired. Ask the user to provide another card."
            )
            return None
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
                "The expiration date has been updated.\n"
                "Do not repeat the expiration date back to the user, ask them to repeat themselves.\n"
                "Call `confirm_expiration_date` once the user confirms, do not call it preemptively.\n"
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
        extra_instructions: str = "",
    ) -> None:
        super().__init__(
            instructions="*none*",
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools or [],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )
        self._require_confirmation = require_confirmation
        self._extra_instructions = extra_instructions

    async def on_enter(self) -> None:
        # Pass chat_ctx into both the TaskGroup AND every sub-task. The
        # TaskGroup overwrites each sub-task's chat_ctx with its own (see
        # TaskGroup.on_enter) - without seeding the TaskGroup, sub-tasks
        # would run with empty context.
        ctx = self.chat_ctx
        # Role hint for the cardholder sub-task. With IGNORE_ON_ENTER on
        # update_name (via require_explicit_ask=True), the model is
        # structurally forced to ask before recording. The extra text
        # just makes sure the *question* anchors to the card.
        cardholder_extra = (
            "You are collecting the name on the credit card (the cardholder). "
            "When you ask the user to confirm a candidate name from earlier in "
            "the conversation, anchor the question to the card or cardholder "
            "so the user knows which name you mean - not just 'is it [name]?' "
            "in the abstract."
        )
        if self._extra_instructions:
            cardholder_extra = f"{self._extra_instructions}\n\n{cardholder_extra}"

        while not self.done():
            # Order: number first (most natural for the caller to give
            # when asked for "card details"), then expiry and security
            # code, then the cardholder name LAST. The name most often
            # pre-fills from chat_ctx (same as the booking name) so
            # leaving it for the end avoids the failure mode where the
            # caller's first response (typically the digits) gets crammed
            # into update_name.
            task_group = TaskGroup(chat_ctx=ctx)
            task_group.add(
                lambda: GetCardNumberTask(
                    chat_ctx=ctx,
                    require_confirmation=self._require_confirmation,
                    extra_instructions=self._extra_instructions,
                ),
                id="card_number_task",
                description="Collects the user's card number",
            )
            task_group.add(
                lambda: GetExpirationDateTask(
                    chat_ctx=ctx,
                    require_confirmation=self._require_confirmation,
                    extra_instructions=self._extra_instructions,
                ),
                id="expiration_date_task",
                description="Collects the card's expiration date",
            )
            task_group.add(
                lambda: GetSecurityCodeTask(
                    chat_ctx=ctx,
                    require_confirmation=self._require_confirmation,
                    extra_instructions=self._extra_instructions,
                ),
                id="security_code_task",
                description="Collects the card's security code",
            )
            task_group.add(
                lambda: GetNameTask(
                    last_name=True,
                    chat_ctx=ctx,
                    extra_instructions=cardholder_extra,
                    require_confirmation=self._require_confirmation,
                    # The cardholder may differ from the caller or any guest
                    # mentioned earlier in chat_ctx. Apply IGNORE_ON_ENTER on
                    # update_name so the model must produce an asking turn
                    # rather than silently filling from chat_ctx.
                    require_explicit_ask=True,
                ),
                id="cardholder_name_task",
                description="Collects the cardholder's full name",
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
