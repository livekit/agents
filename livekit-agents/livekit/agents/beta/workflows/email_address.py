from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask

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
                "Always call `update_email_address` immediately whenever you form a new hypothesis about the email. (before asking any questions or providing any answers.) \n"
                "Call `confirm_email_address` **only** after explicitly asking the user to confirm that the provided email address is correct. \n"
                "If the email is unclear or invalid, prompt for it in parts—first the part before the '@', then the domain—only if needed. \n"
                "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary."
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

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Ask the user to provide their email address. If you already have it, ask for confirmation."
        )

    @function_tool
    async def update_email_address(self, email: str) -> str:
        """Store your best guess of the user's email address.
        This must be called at the earliest opportunity

        Args:
            email: The corrected email address provided by the language model.
        """
        email = email.strip()

        if not re.match(EMAIL_REGEX, email):
            raise ToolError(f"Invalid email address provided: {email}")

        self._current_email = email
        separated_email = " ".join(email)

        return (
            f"The email has been updated: {email}\n"
            f"For clarity, you can also repeat it character by character: {separated_email} if needed"
        )

    @function_tool
    async def confirm_email_address(self) -> None:
        """Validates the email address only after the user has explicitly confirmed it.
        Always prompt the user for confirmation before calling this function."""
        if not self._current_email.strip():
            raise ToolError(
                "no email address were provided, `update_email_address` must be called at least once before calling `confirm_email_address`"
            )

        self.complete(GetEmailResult(email_address=self._current_email))

    @function_tool
    async def decline_email_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide an email address.

        Args:
            reason: A short explanation of why the user declined
        """
        self.complete(ToolError(f"failed to get the user's email address: {reason}"))
