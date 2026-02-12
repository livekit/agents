from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode

PHONE_REGEX = r"^\+?[1-9]\d{6,14}$"


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
        super().__init__(
            instructions=(
                "You are only a single step in a broader system, responsible solely for capturing a phone number.\n"
                "Handle input as noisy voice transcription. Expect that users will say phone numbers aloud with formats like:\n"
                "- '555 123 4567'\n"
                "- 'five five five, one two three, four five six seven'\n"
                "- '+1 555 123 4567'\n"
                "- 'area code 555, 123 4567'\n"
                "- '555-123-4567'\n"
                "Normalize common spoken patterns silently:\n"
                "- Convert spoken digits to their numeric form: 'five' → 5, 'zero' → 0, 'oh' → 0.\n"
                "- Remove filler words, pauses, and hesitations.\n"
                "- Strip dashes, spaces, parentheses, and dots from the number.\n"
                "- Recognize 'plus' at the start as the international prefix `+`.\n"
                "- Recognize 'area code' as a prefix for the area code digits.\n"
                "Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.\n"
                "Call `update_phone_number` at the first opportunity whenever you form a new hypothesis about the phone number. "
                "(before asking any questions or providing any answers.)\n"
                "Don't invent phone numbers, stick strictly to what the user said.\n"
                + (
                    "Call `confirm_phone_number` after the user confirmed the phone number is correct.\n"
                    if require_confirmation is not False
                    else ""
                )
                + "If the number is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the area code, then the remaining digits.\n"
                "Never repeat the phone number back to the user as a single block of digits. Read it back in groups.\n"
                "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.\n"
                "Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."
                + extra_instructions
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

    def _build_confirm_tool(self, *, phone_number: str):
        # confirm tool is only injected after update_phone_number is called,
        # preventing the LLM from hallucinating a confirmation without user input
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
