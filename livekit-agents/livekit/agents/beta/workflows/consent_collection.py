from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask

if TYPE_CHECKING:
    from ...voice.agent_session import TurnDetectionMode


@dataclass
class CollectConsentResult:
    consent: bool
    denied_reason: str | None


class CollectConsentTask(AgentTask[CollectConsentResult]):
    def __init__(
        self,
        *,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        extra_instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        instructions = self.base_instructions
        if extra_instructions:
            instructions += "\n" + extra_instructions
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

    @property
    def base_instructions(self) -> str:
        return (
            "You are responsible for collecting consent from the user. "
            "Call `consent_given` when the user explicitly gives consent or shows interest to follow up. "
            "Call `consent_denied` when the user denies consent. "
            "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary. \n"
            "Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Ask the user to provide their consent.", tool_choice="none"
        )

    @function_tool
    async def consent_given(self) -> None:
        """Called when the user explicitly gives consent or shows interest to follow up."""
        if not self.done():
            self.complete(CollectConsentResult(consent=True, denied_reason=None))

    @function_tool
    async def consent_denied(self, reason: str) -> None:
        """Called when the user denies consent.

        Args:
            reason: The reason why the user denied consent, "unknown" if not stated
        """
        if not self.done():
            self.complete(CollectConsentResult(consent=False, denied_reason=reason))
