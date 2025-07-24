from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from ...voice.speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ...voice.agent_session import TurnDetectionMode

EMAIL_REGEX = (
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)


@dataclass
class GetEmailResult:
    email_address: str


class GetEmailTask(AgentTask[GetEmailResult]):
    def __init__(
        self,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(
                "You are only a single step in a broader system, responsible solely for capturing an email address.\n"
                "Handle input as noisy voice transcription. Expect that users will say emails aloud with formats like:\n"
                "- 'john dot doe at gmail dot com'\n"
                "- 'susan underscore smith at yahoo dot co dot uk'\n"
                "- 'dave dash b at protonmail dot com'\n"
                "- 'jane at example' (partial—prompt for the domain)\n"
                "- 'theo t h e o at livekit dot io' (name followed by spelling)\n"
                "Normalize common spoken patterns silently:\n"
                "- Convert words like 'dot', 'underscore', 'dash', 'plus' into symbols: `.`, `_`, `-`, `+`.\n"
                "- Convert 'at' to `@`.\n"
                "- Recognize patterns where users speak their name or a word, followed by spelling: e.g., 'john j o h n'.\n"
                "- Filter out filler words or hesitations.\n"
                "- Assume some spelling if contextually obvious (e.g. 'mike b two two' → mikeb22).\n"
                "Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.\n"
                "Call `update_email_address` at the first opportunity whenever you form a new hypothesis about the email. "
                "(before asking any questions or providing any answers.) \n"
                "Don't invent new email addresses, stick strictly to what the user said. \n"
                "Call `confirm_email_address` after the user confirmed the email address is correct. \n"
                "If the email is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the part before the '@', then the domain—only if needed. \n"
                "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary. \n"
                "Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._current_email = ""

        # speech_handle/turn used to update the email address.
        # used to ignore the call to confirm_email_address in case the LLM is hallucinating and not asking for user confirmation
        self._email_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Ask the user to provide an email address. If you already have it, ask for confirmation.\n"
                "Do not call `decline_email_capture`"
            )
        )

    @function_tool
    async def update_email_address(self, email: str, ctx: RunContext) -> str:
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

        return (
            f"The email has been updated to {email}\n"
            f"Repeat the email character by character: {separated_email} if needed\n"
            f"Prompt the user for confirmation, do not call `confirm_email_address` directly"
        )

    @function_tool
    async def confirm_email_address(self, ctx: RunContext) -> None:
        """Validates/confirms the email address provided by the user."""
        await ctx.wait_for_playout()

        if ctx.speech_handle == self._email_update_speech_handle:
            raise ToolError("error: the user must confirm the email address explicitly")

        if not self._current_email.strip():
            raise ToolError(
                "error: no email address were provided, `update_email_address` must be called before"
            )

        if not self.done():
            self.complete(GetEmailResult(email_address=self._current_email))

    @function_tool
    async def decline_email_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide an email address.

        Args:
            reason: A short explanation of why the user declined to provide the email address
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the email address: {reason}"))
