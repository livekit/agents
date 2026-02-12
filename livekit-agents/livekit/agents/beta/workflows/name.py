from __future__ import annotations

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
        name_format: NotGivenOr[str] = NOT_GIVEN,
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
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if not (first_name or middle_name or last_name):
            raise ValueError("At least one of first_name, middle_name, or last_name must be True")
        self._collect_first_name = first_name
        self._collect_last_name = last_name
        self._collect_middle_name = middle_name
        self._verify_spelling = verify_spelling
        self._require_confirmation = require_confirmation

        if is_given(name_format):
            self._name_format = name_format
        else:
            parts = []
            if first_name:
                parts.append("{first_name}")
            if middle_name:
                parts.append("{middle_name}")
            if last_name:
                parts.append("{last_name}")
            self._name_format = " ".join(parts)

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
                f"You need to collect the name parts in this order: {self._name_format}.\n"
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
                + (
                    "Call `confirm_name` after the user confirmed the name is correct.\n"
                    if require_confirmation is not False
                    else ""
                )
                + "If the name is unclear or it takes too much back-and-forth, prompt for each name part separately.\n"
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

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=f"Ask the user to provide their name in this format: {self._name_format}."
        )

    @function_tool()
    async def update_name(
        self,
        first_name: str,
        ctx: RunContext,
        middle_name: str | None = None,
        last_name: str | None = None,
    ) -> str | None:
        """Update the name provided by the user.

        Args:
            first_name: The user's first name.
            middle_name: The user's middle name, if collected.
            last_name: The user's last name, if collected.
        """
        errors: list[str] = []
        if self._collect_first_name and not (first_name and first_name.strip()):
            errors.append("first name is required but was not provided")
        if self._collect_middle_name and not (middle_name and middle_name.strip()):
            errors.append("middle name is required but was not provided")
        if self._collect_last_name and not (last_name and last_name.strip()):
            errors.append("last name is required but was not provided")

        if errors:
            raise ToolError(f"Incomplete name: {'; '.join(errors)}")

        self._first_name = first_name.strip() if first_name else ""
        self._middle_name = middle_name.strip() if middle_name else ""
        self._last_name = last_name.strip() if last_name else ""

        full_name = self._name_format.format(
            first_name=self._first_name,
            middle_name=self._middle_name,
            last_name=self._last_name,
        ).strip()

        if not self._confirmation_required(ctx):
            if not self.done():
                self.complete(
                    GetNameResult(
                        first_name=self._first_name if self._collect_first_name else None,
                        middle_name=self._middle_name if self._collect_middle_name else None,
                        last_name=self._last_name if self._collect_last_name else None,
                    )
                )
            return None

        confirm_tool = self._build_confirm_tool(
            first_name=self._first_name,
            middle_name=self._middle_name,
            last_name=self._last_name,
        )
        current_tools = [t for t in self.tools if t.id != "confirm_name"]
        current_tools.append(confirm_tool)
        await self.update_tools(current_tools)

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

    def _build_confirm_tool(
        self, *, first_name: str, middle_name: str, last_name: str
    ) -> llm.FunctionTool:
        @function_tool()
        async def confirm_name() -> None:
            """Call after the user confirms the name is correct."""
            if (
                first_name != self._first_name
                or middle_name != self._middle_name
                or last_name != self._last_name
            ):
                self.session.generate_reply(
                    instructions="The name has changed since confirmation was requested, ask the user to confirm the updated name."
                )
                return

            if not self.done():
                self.complete(
                    GetNameResult(
                        first_name=self._first_name if self._collect_first_name else None,
                        middle_name=self._middle_name if self._collect_middle_name else None,
                        last_name=self._last_name if self._collect_last_name else None,
                    )
                )

        return confirm_name

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_name_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide their name.

        Args:
            reason: A short explanation of why the user declined to provide their name
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the name: {reason}"))

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"
