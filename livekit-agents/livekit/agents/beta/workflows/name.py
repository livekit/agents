from __future__ import annotations

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

_BASE_INSTRUCTIONS = """
You are only a single step in a broader system, responsible solely for capturing the user's name.
You need to naturally collect the name parts in this order: {name_format}.
{modality_specific}
{spelling_instructions}Call `update_name` at the first opportunity whenever you form a new hypothesis about the name. (before asking any questions or providing any answers.)
Don't invent names, stick strictly to what the user said.
{confirmation_instructions}
If the name is unclear or it takes too much back-and-forth, prompt for each name part separately.
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Avoid verbosity by not sharing example names or spellings unless prompted to do so. Do not deviate from the goal of collecting the user's name.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.\
{extra_instructions}
"""

_AUDIO_SPECIFIC = """
Handle input as noisy voice transcription. Expect that users will say names aloud and may:
- Say their name followed by spelling: e.g., 'Michael m i c h a e l'
- Use phonetic alphabet: e.g., 'Mike as in Mike India Charlie Hotel Alpha Echo Lima'
- Have names with special characters or hyphens: e.g., 'Mary-Jane' or 'O'Brien'
- Have names from various cultural backgrounds with different pronunciation patterns
Normalize common spoken patterns silently:
- Convert 'dash' or 'hyphen' to `-`.
- Convert 'apostrophe' to `'`.
- Recognize when users spell out their name letter by letter.
- Filter out filler words or hesitations.
- Capitalize the first letter of each name part appropriately.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
"""

_TEXT_SPECIFIC = """
Handle input as typed text. Expect users to type their name directly.
Capitalize the first letter of each name part appropriately.
If the name contains special characters or hyphens (e.g., 'Mary-Jane' or 'O'Brien'), preserve them as typed.
"""


def _clean_name_arg(value: str | None) -> str | None:
    # Some models (e.g. gemma) fill optional args with placeholder strings like
    # "null"/"NULL" instead of omitting them, or wrap values in literal quotes.
    # Normalize those to None/clean values so they hit the required-field
    # validation below instead of being recorded as the user's name.
    if value is None:
        return None
    value = value.strip().strip("'\"")
    if not value or value.casefold() in ("null", "none", "nil", "n/a", "unknown", "unspecified"):
        return None
    return value


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
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
        require_explicit_ask: bool = False,
    ) -> None:
        if not (first_name or middle_name or last_name):
            raise ValueError("At least one of first_name, middle_name, or last_name must be True")
        self._collect_first_name = first_name
        self._collect_last_name = last_name
        self._collect_middle_name = middle_name
        self._verify_spelling = verify_spelling
        self._require_confirmation = require_confirmation
        self._require_explicit_ask = require_explicit_ask

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
        confirmation_instructions = (
            "Call `confirm_name` after the user confirmed the name is correct."
        )
        extra = extra_instructions if extra_instructions else ""

        # State initialized BEFORE super() so the dynamically-built
        # update_name tool's closures see real attrs at call time.
        self._first_name: str = ""
        self._middle_name: str = ""
        self._last_name: str = ""

        super().__init__(
            instructions=Instructions(
                _BASE_INSTRUCTIONS.format(
                    name_format=self._name_format,
                    modality_specific=_AUDIO_SPECIFIC,
                    spelling_instructions=spelling_instructions,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is not False else ""
                    ),
                    extra_instructions=extra,
                ),
                text=_BASE_INSTRUCTIONS.format(
                    name_format=self._name_format,
                    modality_specific=_TEXT_SPECIFIC,
                    spelling_instructions=spelling_instructions,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is True else ""
                    ),
                    extra_instructions=extra,
                ),
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=[*(tools or []), self._build_update_name_tool()],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                f"Get the user's name (follow this order '{self._name_format}' but do not "
                "mention the format). First scan the conversation - if a name was already "
                "given earlier, ask a short confirmation question rather than asking from "
                "scratch. If context about what the name is FOR was provided (a role like "
                "'cardholder', 'guest', 'emergency contact'), anchor your confirmation "
                "question to that role so the user knows which name you mean - don't ask "
                "abstractly. When pointing at where an existing name came from, reference "
                "the source in the conversation (the earlier step, the booking they "
                "mentioned), not a presumption about how the name appears in the "
                "destination. Only ask fresh when the conversation has no name yet."
            )
        )

    def _build_update_name_tool(self) -> llm.FunctionTool:
        # Built dynamically so we can apply IGNORE_ON_ENTER per-instance
        # based on require_explicit_ask. With the flag set, the model
        # can't silent-fill from chat_ctx during on_enter - it must
        # produce an asking utterance first.
        flags = ToolFlag.IGNORE_ON_ENTER if self._require_explicit_ask else ToolFlag.NONE

        @function_tool(flags=flags)
        async def update_name(
            ctx: RunContext,
            first_name: str | None = None,
            middle_name: str | None = None,
            last_name: str | None = None,
        ) -> str | None:
            """Update the name provided by the user.

            Args:
                first_name: The user's first name.
                middle_name: The user's middle name, if collected.
                last_name: The user's last name, if collected.
            """
            return await self._update_name_impl(ctx, first_name, middle_name, last_name)

        return update_name

    async def _update_name_impl(
        self,
        ctx: RunContext,
        first_name: str | None = None,
        middle_name: str | None = None,
        last_name: str | None = None,
    ) -> str | None:
        first_name, middle_name, last_name = (
            _clean_name_arg(v) for v in (first_name, middle_name, last_name)
        )
        errors: list[str] = []
        if self._collect_first_name and not (first_name and first_name.strip()):
            errors.append("first name is required but was not provided")
        if self._collect_middle_name and not (middle_name and middle_name.strip()):
            errors.append("middle name is required but was not provided")
        if self._collect_last_name and not (last_name and last_name.strip()):
            errors.append("last name is required but was not provided")

        # A real name contains letters. Reject digit-only or punctuation-only
        # values so a card number, ZIP code, phone number, etc. accidentally
        # crammed into update_name fails fast instead of being recorded as
        # the user's name.
        for label, value in (
            ("first", first_name),
            ("middle", middle_name),
            ("last", last_name),
        ):
            if value and value.strip() and not any(c.isalpha() for c in value):
                errors.append(
                    f"{label} name {value!r} contains no letters - that doesn't look like a name"
                )

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
