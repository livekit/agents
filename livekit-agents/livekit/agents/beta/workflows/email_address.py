from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm import Instructions
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...log import logger
from ...types import NOT_GIVEN, NotGiven, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from ...voice.speech_handle import SpeechHandle
from .utils import InstructionParts, build_instructions

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode

EMAIL_REGEX = (
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)

# instructions
ROLE = "You are only a single step in a broader system, responsible solely for capturing an email address."

CONTEXT = Instructions(
    audio="""\
Handle input as noisy voice transcription. Expect that users will say emails aloud with formats like:
- 'john dot doe at gmail dot com'
- 'susan underscore smith at yahoo dot co dot uk'
- 'dave dash b at protonmail dot com'
- 'jane at example' (partial—prompt for the domain)
- 'theo t h e o at livekit dot io' (name followed by spelling)
Normalize common spoken patterns silently:
- Convert words like 'dot', 'underscore', 'dash', 'plus' into symbols: `.`, `_`, `-`, `+`.
- Convert 'at' to `@`.
- Recognize patterns where users speak their name or a word, followed by spelling: e.g., 'john j o h n'.
- Filter out filler words or hesitations.
- Assume some spelling if contextually obvious (e.g. 'mike b two two' → mikeb22).
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.""",
    text="""\
Handle input as typed text. Expect users to type their email address directly in standard format.
If the address looks almost correct but has minor typos (e.g. missing '@' or domain), prompt for clarification.""",
)

CONSTRAINTS = """\
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."""

# internal directive, coupled to tool names, not user-customizable.
_DIRECTIVE = """\
Call `update_email_address` at the first opportunity whenever you form a new hypothesis about the email. (before asking any questions or providing any answers.)
Don't invent new email addresses, stick strictly to what the user said.
{confirmation}
If the email is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the part before the '@', then the domain—only if needed."""

_CONFIRMATION_INSTRUCTION = (
    "Call `confirm_email_address` after the user confirmed the email address is correct."
)


@dataclass
class GetEmailResult:
    email_address: str


class GetEmailTask(AgentTask[GetEmailResult]):
    INSTRUCTION_PARTS = InstructionParts(role=ROLE, context=CONTEXT, constraints=CONSTRAINTS)

    def __init__(
        self,
        *,
        instructions: NotGivenOr[InstructionParts | Instructions | str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
        # deprecated
        extra_instructions: str = "",
    ) -> None:
        if instructions and extra_instructions:
            logger.warning("`extra_instructions` will be ignored when `instructions` is provided")

        if isinstance(instructions, InstructionParts | NotGiven):
            directive = Instructions(
                audio=_DIRECTIVE.format(
                    confirmation=_CONFIRMATION_INSTRUCTION
                    if require_confirmation is not False  # enabled by default
                    else ""
                ),
                text=_DIRECTIVE.format(
                    confirmation=_CONFIRMATION_INSTRUCTION
                    if require_confirmation is True  # disabled by default
                    else ""
                ),
            )
            instructions = build_instructions(
                parts=instructions or InstructionParts(extra=extra_instructions),
                defaults=self.INSTRUCTION_PARTS,
                directive=directive,
            )

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools or [],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._current_email = ""
        self._require_confirmation = require_confirmation
        # speech_handle/turn used to update the email address.
        # used to ignore the call to confirm_email_address in case the LLM is hallucinating and not asking for user confirmation
        self._email_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask the user to provide an email address.")

    @function_tool
    async def update_email_address(self, email: str, ctx: RunContext) -> str | None:
        """Update the email address provided by the user.

        Args:
            email: The email address provided by the user
        """
        self._email_update_speech_handle = ctx.speech_handle
        email = email.strip()

        if not re.match(EMAIL_REGEX, email):
            raise ToolError(f"Invalid email address provided: {email}")

        self._current_email = email
        separated_email = " ".join(email)

        if not self._confirmation_required(ctx):
            if not self.done():
                self.complete(GetEmailResult(email_address=self._current_email))
            return None  # no need to continue the conversation

        return (
            f"The email has been updated to {email}\n"
            f"Repeat the email character by character: {separated_email} if needed\n"
            f"Prompt the user for confirmation, do not call `confirm_email_address` directly"
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def confirm_email_address(self, ctx: RunContext) -> None:
        """Validates/confirms the email address provided by the user."""
        await ctx.wait_for_playout()

        if ctx.speech_handle == self._email_update_speech_handle and self._confirmation_required(
            ctx
        ):
            raise ToolError("error: the user must confirm the email address explicitly")

        if not self._current_email.strip():
            raise ToolError(
                "error: no email address were provided, `update_email_address` must be called before"
            )

        if not self.done():
            self.complete(GetEmailResult(email_address=self._current_email))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_email_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide an email address.

        Args:
            reason: A short explanation of why the user declined to provide the email address
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the email address: {reason}"))

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"
