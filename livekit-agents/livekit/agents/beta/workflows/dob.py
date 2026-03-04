from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext

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
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
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
                + (
                    "Call `confirm_dob` after the user confirmed the date of birth is correct.\n"
                    if require_confirmation is not False
                    else ""
                )
                + "When reading back dates, use a natural spoken format like 'January fifteenth, nineteen ninety'.\n"
                "If the date is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the month, then the day, then the year.\n"
                "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.\n"
                "Avoid verbosity by not sharing example dates or formats unless prompted to do so. Do not deviate from the goal of collecting the user's birthday.\n"
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

        self._include_time = include_time
        self._require_confirmation = require_confirmation
        self._current_dob: date | None = None
        self._current_time: time | None = None

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
    ) -> str | None:
        """Update the date of birth provided by the user. Given a spoken month and year (e.g., 'July 2030'), return its numerical representation (7/2030).

        Args:
            year: The birth year (e.g., 1990)
            month: The birth month (1-12)
            day: The birth day (1-31)
        """
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

        if not self._confirmation_required(ctx):
            if not self.done():
                self.complete(
                    GetDOBResult(
                        date_of_birth=self._current_dob,
                        time_of_birth=self._current_time,
                    )
                )
            return None

        confirm_tool = self._build_confirm_tool(dob=dob)
        current_tools = [t for t in self.tools if t.id != "confirm_dob"]
        current_tools.append(confirm_tool)
        await self.update_tools(current_tools)

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
    ) -> str | None:
        """Update the time of birth provided by the user.

        Args:
            hour: The birth hour (0-23)
            minute: The birth minute (0-59)
        """
        try:
            birth_time = time(hour, minute)
        except ValueError as e:
            raise ToolError(f"Invalid time: {e}") from None

        self._current_time = birth_time

        if not self._confirmation_required(ctx) and self._current_dob is not None:
            if not self.done():
                self.complete(
                    GetDOBResult(
                        date_of_birth=self._current_dob,
                        time_of_birth=self._current_time,
                    )
                )
            return None

        if self._confirmation_required(ctx):
            confirm_tool = self._build_confirm_tool(dob=self._current_dob)
            current_tools = [t for t in self.tools if t.id != "confirm_dob"]
            current_tools.append(confirm_tool)
            await self.update_tools(current_tools)

        formatted_time = birth_time.strftime("%I:%M %p")
        response = f"The time of birth has been updated to {formatted_time}"

        if self._current_dob:
            formatted_date = self._current_dob.strftime("%B %d, %Y")
            response = f"The date and time of birth has been updated to {formatted_date} at {formatted_time}"

        if self._confirmation_required(ctx):
            response += (
                "\nRepeat the time back to the user in a natural spoken format.\n"
                "Prompt the user for confirmation, do not call `confirm_dob` directly"
            )
        else:
            response += "\nThe date of birth has not been provided yet, ask the user to provide it."

        return response

    def _build_confirm_tool(self, *, dob: date | None) -> llm.FunctionTool:
        # confirm tool is only injected after update_dob/update_time is called,
        # preventing the LLM from hallucinating a confirmation without user input
        captured_dob = dob
        captured_time = self._current_time

        @function_tool()
        async def confirm_dob() -> None:
            """Call after the user confirms the date of birth is correct."""
            if captured_dob != self._current_dob or captured_time != self._current_time:
                self.session.generate_reply(
                    instructions="The date of birth has changed since confirmation was requested, ask the user to confirm the updated date."
                )
                return

            if self._current_dob is None:
                self.session.generate_reply(
                    instructions="No date of birth was provided yet, ask the user to provide it."
                )
                return

            if not self.done():
                self.complete(
                    GetDOBResult(
                        date_of_birth=self._current_dob,
                        time_of_birth=self._current_time,
                    )
                )

        return confirm_dob

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_dob_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide a date of birth.

        Args:
            reason: A short explanation of why the user declined to provide the date of birth
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the date of birth: {reason}"))

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"
