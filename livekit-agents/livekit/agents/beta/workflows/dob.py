from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from ...voice.speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode


@dataclass
class GetDOBResult:
    date_of_birth: date
    time_of_birth: time | None = None


class GetDOBTask(AgentTask[GetDOBResult]):
    def __init__(
        self,
        extra_instructions: str = "",
        include_time: bool = False,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        time_instructions = (
            ""
            if not include_time
            else (
                "Also ask for and capture the time of birth if the user knows it. "
                "The time is optional - if the user doesn't know it, proceed without it.\n"
            )
        )

        super().__init__(
            instructions=(
                "You are only a single step in a broader system, responsible solely for capturing a date of birth.\n"
                "Handle input as noisy voice transcription. Expect that users will say dates aloud with formats like:\n"
                "- 'January 15th 1990'\n"
                "- 'the fifteenth of January nineteen ninety'\n"
                "- '01 15 1990' or 'one fifteen ninety'\n"
                "- 'Jan 15 90'\n"
                "- '15th January 1990'\n"
                "Normalize common spoken patterns silently:\n"
                "- Convert spoken numbers and ordinals to their numeric form: 'fifteenth' → 15, 'ninety' → 1990.\n"
                "- Recognize month names in various forms: 'Jan', 'January', etc.\n"
                "- Handle two-digit years appropriately: '90' likely means 1990, '05' likely means 2005.\n"
                "- Filter out filler words or hesitations.\n"
                "Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.\n"
                f"{time_instructions}"
                "Call `update_dob` at the first opportunity whenever you form a new hypothesis about the date of birth. "
                "(before asking any questions or providing any answers.)\n"
                "Don't invent dates, stick strictly to what the user said.\n"
                "Call `confirm_dob` after the user confirmed the date of birth is correct.\n"
                "When reading back dates, use a natural spoken format like 'January fifteenth, nineteen ninety'.\n"
                "If the date is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the month, then the day, then the year.\n"
                "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.\n"
                "Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."
                + extra_instructions
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._include_time = include_time
        self._current_dob: date | None = None
        self._current_time: time | None = None
        self._dob_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        prompt = "Ask the user to provide their date of birth."
        if self._include_time:
            prompt = "Ask the user to provide their date of birth and, if they know it, their time of birth."
        self.session.generate_reply(instructions=prompt)

    @function_tool
    async def update_dob(
        self,
        year: int,
        month: int,
        day: int,
        ctx: RunContext,
    ) -> str:
        """Update the date of birth provided by the user.

        Args:
            year: The birth year (e.g., 1990)
            month: The birth month (1-12)
            day: The birth day (1-31)
        """
        self._dob_update_speech_handle = ctx.speech_handle

        try:
            dob = date(year, month, day)
        except ValueError as e:
            raise ToolError(f"Invalid date: {e}") from None

        today = date.today()
        if dob > today:
            raise ToolError(
                f"Invalid date of birth: {dob.strftime('%B %d, %Y')} is in the future. "
                "Date of birth cannot be a future date."
            )

        self._current_dob = dob

        formatted_date = dob.strftime("%B %d, %Y")
        response = f"The date of birth has been updated to {formatted_date}"

        if self._current_time:
            formatted_time = self._current_time.strftime("%I:%M %p")
            response += f" at {formatted_time}"

        response += (
            "\nRepeat the date back to the user in a natural spoken format.\n"
            "Prompt the user for confirmation, do not call `confirm_dob` directly"
        )

        return response

    @function_tool
    async def update_time(
        self,
        hour: int,
        minute: int,
        ctx: RunContext,
    ) -> str:
        """Update the time of birth provided by the user.

        Args:
            hour: The birth hour (0-23)
            minute: The birth minute (0-59)
        """
        self._dob_update_speech_handle = ctx.speech_handle

        try:
            birth_time = time(hour, minute)
        except ValueError as e:
            raise ToolError(f"Invalid time: {e}") from None

        self._current_time = birth_time

        formatted_time = birth_time.strftime("%I:%M %p")
        response = f"The time of birth has been updated to {formatted_time}"

        if self._current_dob:
            formatted_date = self._current_dob.strftime("%B %d, %Y")
            response = f"The date and time of birth has been updated to {formatted_date} at {formatted_time}"

        response += (
            "\nRepeat the time back to the user in a natural spoken format.\n"
            "Prompt the user for confirmation, do not call `confirm_dob` directly"
        )

        return response

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def confirm_dob(self, ctx: RunContext) -> None:
        """Call this tool when the user confirms that the date of birth is correct."""
        await ctx.wait_for_playout()

        if ctx.speech_handle == self._dob_update_speech_handle:
            raise ToolError("error: the user must confirm the date of birth explicitly")

        if self._current_dob is None:
            raise ToolError(
                "error: no date of birth was provided, `update_dob` must be called before"
            )

        if not self.done():
            self.complete(
                GetDOBResult(
                    date_of_birth=self._current_dob,
                    time_of_birth=self._current_time,
                )
            )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_dob_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide a date of birth.

        Args:
            reason: A short explanation of why the user declined to provide the date of birth
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the date of birth: {reason}"))
