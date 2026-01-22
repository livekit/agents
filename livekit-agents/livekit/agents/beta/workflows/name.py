from __future__ import annotations

from dataclasses import dataclass
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
class GetNameResult:
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None


class GetNameTask(AgentTask[GetNameResult]):
    def __init__(
        self,
        first_name: bool = True,
        last_name: bool = False,
        middle_name: bool = False,
        verify_spelling: bool = False,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if not (first_name or middle_name or last_name):
            raise ValueError("At least one of first_name, middle_name, or last_name must be True")
        self._collect_first_name = first_name
        self._collect_last_name = last_name
        self._collect_middle_name = middle_name
        self._verify_spelling = verify_spelling

        self._requested_name_parts: list[
            str
        ] = []  # this builds a string to pass into the LLM instructions
        if first_name:
            self._requested_name_parts.append("first name")
        if middle_name:
            self._requested_name_parts.append("middle name")
        if last_name:
            self._requested_name_parts.append("last name")

        spelling_instructions = (
            ""
            if not verify_spelling
            else (
                "After receiving the name, always verify the spelling by asking the user to confirm "
                "or spell out the name letter by letter. "
                "When confirming, spell out each name part letter by letter to the user. "
            )
        )

        super().__init__(
            instructions=(
                f"You are only a single step in a broader system, responsible solely for capturing the user's name.\n"
                f"You need to collect the following name parts: {', '.join(self._requested_name_parts)}.\n"
                "Handle input as noisy voice transcription. Expect that users will say names aloud and may:\n"
                "- Say their name followed by spelling: e.g., 'Michael m i c h a e l'\n"
                "- Use phonetic alphabet: e.g., 'Mike as in Mike India Charlie Hotel Alpha Echo Lima'\n"
                "- Have names with special characters or hyphens: e.g., 'Mary-Jane' or 'O'Brien'\n"
                "- Have names from various cultural backgrounds with different pronunciation patterns\n"
                "Normalize common spoken patterns silently:\n"
                "- Convert 'dash' or 'hyphen' to `-`.\n"
                "- Convert 'apostrophe' to `'`.\n"
                "- Recognize when users spell out their name letter by letter.\n"
                "- Filter out filler words or hesitations.\n"
                "- Capitalize the first letter of each name part appropriately.\n"
                "Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.\n"
                f"{spelling_instructions}"
                "Call `update_name` at the first opportunity whenever you form a new hypothesis about the name. "
                "(before asking any questions or providing any answers.)\n"
                "Don't invent names, stick strictly to what the user said.\n"
                "Call `confirm_name` after the user confirmed the name is correct.\n"
                "If the name is unclear or it takes too much back-and-forth, prompt for each name part separately.\n"
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

        self._first_name: str = ""
        self._middle_name: str = ""
        self._last_name: str = ""

        self._name_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        if len(self._requested_name_parts) == 1:
            prompt = f"Ask the user to provide their {self._requested_name_parts[0]}."
        else:
            prompt = f"Ask the user to provide their {', '.join(self._requested_name_parts)}."

        self.session.generate_reply(instructions=prompt)

    @function_tool()
    async def update_name(
        self,
        first_name: str,
        middle_name: str,
        last_name: str,
        ctx: RunContext,
    ) -> str:
        """Update the name provided by the user.

        Args:
            first_name: The user's first name. Use empty string if not collected or not applicable.
            middle_name: The user's middle name. Use empty string if not collected or not applicable.
            last_name: The user's last name. Use empty string if not collected or not applicable.
        """
        self._name_update_speech_handle = ctx.speech_handle

        errors: list[str] = []
        if self._collect_first_name and not first_name.strip():
            errors.append("first name is required but was not provided")
        if self._collect_middle_name and not middle_name.strip():
            errors.append("middle name is required but was not provided")
        if self._collect_last_name and not last_name.strip():
            errors.append("last name is required but was not provided")

        if errors:
            raise ToolError(f"Incomplete name: {'; '.join(errors)}")

        self._first_name = first_name.strip()
        self._middle_name = middle_name.strip()
        self._last_name = last_name.strip()

        full_name = " ".join(
            part for part in [self._first_name, self._middle_name, self._last_name] if part
        )

        if self._verify_spelling:
            return (
                f"The name has been updated to {full_name}\n"
                f"Spell out the name letter by letter for verification: {full_name}\n"
                f"Prompt the user for confirmation, do not call `confirm_name` directly"
            )

        return (
            f"The name has been updated to {full_name}\n"
            f"Repeat the name back to the user and prompt for confirmation, "
            f"do not call `confirm_name` directly"
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def confirm_name(self, ctx: RunContext) -> None:
        """Call this tool when the user confirms that the name is correct."""
        await ctx.wait_for_playout()

        if ctx.speech_handle == self._name_update_speech_handle:
            raise ToolError("error: the user must confirm the name explicitly")

        if self._collect_first_name and not self._first_name:
            raise ToolError(
                "error: first name was not provided, `update_name` must be called first"
            )
        if self._collect_middle_name and not self._middle_name:
            raise ToolError(
                "error: middle name was not provided, `update_name` must be called first"
            )
        if self._collect_last_name and not self._last_name:
            raise ToolError("error: last name was not provided, `update_name` must be called first")

        if not self.done():
            self.complete(
                GetNameResult(
                    first_name=self._first_name if self._collect_first_name else None,
                    middle_name=self._middle_name if self._collect_middle_name else None,
                    last_name=self._last_name if self._collect_last_name else None,
                )
            )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_name_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide their name.

        Args:
            reason: A short explanation of why the user declined to provide their name
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the name: {reason}"))
