from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.chat_context import Instructions
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode

PHONE_REGEX = r"^\+?[1-9]\d{6,14}$"

_BASE_INSTRUCTIONS = """
You are only a single step in a broader system, responsible solely for capturing a phone number.
{modality_specific}
Call `update_phone_number` at the first opportunity whenever you form a new hypothesis about the phone number. (before asking any questions or providing any answers.)
Don't invent phone numbers, stick strictly to what the user said.
{confirmation_instructions}
If the number is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the area code, then the remaining digits.
Never repeat the phone number back to the user as a single block of digits. Read it back in groups.
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Avoid verbosity by not sharing example phone numbers or formats unless prompted to do so. Do not deviate from the goal of collecting the user's phone number.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.\
{extra_instructions}
"""

_AUDIO_SPECIFIC = """
Handle input as noisy voice transcription. Expect that users will say phone numbers aloud with formats like:
- '555 123 4567'
- 'five five five, one two three, four five six seven'
- '+1 555 123 4567'
- 'area code 555, 123 4567'
- '555-123-4567'
Normalize common spoken patterns silently:
- Convert spoken digits to their numeric form: 'five' → 5, 'zero' → 0, 'oh' → 0.
- Remove filler words, pauses, and hesitations.
- Strip dashes, spaces, parentheses, and dots from the number.
- Recognize 'plus' at the start as the international prefix `+`.
- Recognize 'area code' as a prefix for the area code digits.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
"""

_TEXT_SPECIFIC = """
Handle input as typed text. Expect users to type their phone number directly.
Strip dashes, spaces, parentheses, and dots from the number.
If the number looks almost correct but has minor formatting issues, clean it up silently.
"""


@dataclass
class GetPhoneNumberResult:
    phone_number: str


class GetPhoneNumberTask(AgentTask[GetPhoneNumberResult]):
    def __init__(
        self,
        extra_instructions: str = "",
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
        confirmation_instructions = (
            "Call `confirm_phone_number` after the user confirmed the phone number is correct."
        )
        extra = extra_instructions if extra_instructions else ""

        super().__init__(
            instructions=Instructions(
                _BASE_INSTRUCTIONS.format(
                    modality_specific=_AUDIO_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is not False else ""
                    ),
                    extra_instructions=extra,
                ),
                text=_BASE_INSTRUCTIONS.format(
                    modality_specific=_TEXT_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is True else ""
                    ),
                    extra_instructions=extra,
                ),
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools or [],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._current_phone_number = ""
        self._require_confirmation = require_confirmation

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask the user to provide their phone number.")

    @function_tool()
    async def update_phone_number(self, phone_number: str, ctx: RunContext) -> str | None:
        """Update the phone number provided by the user.

        Args:
            phone_number: The phone number provided by the user, digits only with optional leading +
        """
        cleaned = re.sub(r"[\s\-().]+", "", phone_number.strip())

        if not re.match(PHONE_REGEX, cleaned):
            raise ToolError(f"Invalid phone number provided: {phone_number}")

        self._current_phone_number = cleaned

        if not self._confirmation_required(ctx):
            if not self.done():
                self.complete(GetPhoneNumberResult(phone_number=self._current_phone_number))
            return None  # no need to continue the conversation

        confirm_tool = self._build_confirm_tool(phone_number=cleaned)
        current_tools = [t for t in self.tools if t.id != "confirm_phone_number"]
        current_tools.append(confirm_tool)
        await self.update_tools(current_tools)

        return (
            f"The phone number has been updated to {cleaned}\n"
            f"Read the number back to the user in groups.\n"
            f"Prompt the user for confirmation, do not call `confirm_phone_number` directly"
        )

    def _build_confirm_tool(self, *, phone_number: str) -> llm.FunctionTool:
        @function_tool()
        async def confirm_phone_number() -> None:
            """Call after the user confirms the phone number is correct."""
            if phone_number != self._current_phone_number:
                self.session.generate_reply(
                    instructions="The phone number has changed since confirmation was requested, ask the user to confirm the updated number."
                )
                return

            if not self.done():
                self.complete(GetPhoneNumberResult(phone_number=phone_number))

        return confirm_phone_number

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_phone_number_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide a phone number.

        Args:
            reason: A short explanation of why the user declined to provide the phone number
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the phone number: {reason}"))

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"
