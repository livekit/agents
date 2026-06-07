from __future__ import annotations

from dataclasses import dataclass

from persona import COMMON_INSTRUCTIONS

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_CONSENT_INSTRUCTIONS = """\
Your job right now: get the caller's permission to record the call for quality.

The question MUST be phrased so a "yes" answer means "yes, record me". Use affirmative framings: "is that OK?", "alright with you?", "good with that?", "sound OK?". Never use obstacle framings like "do you mind?", "any objection?", "is that a problem?" - those invert the polarity and a one-word "nope" then means the opposite of what it sounds like.

Each turn, do EITHER - never both:
- COMMIT: you're certain of the answer -> call the matching tool. Don't say anything else this turn; the call moves on.
- ASK: the answer is ambiguous, off-topic, or they haven't answered yet -> one short steering line, no tool call. Trust the caller to remember what was just said; keep the steer fresh and brief.
"""


@dataclass
class RecordingConsentResult:
    consent: bool
    declined_reason: str | None = None


class GetRecordingConsentTask(AgentTask[RecordingConsentResult]):
    def __init__(self, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_CONSENT_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the caller in one short sentence and ask permission to record."
        )

    @function_tool()
    async def record_consent(self, consents: bool) -> None:
        """Record the caller's recording-consent answer.

        Args:
            consents: True if the caller agreed to recording; False otherwise.
        """
        if not self.done():
            self.complete(RecordingConsentResult(consent=consents))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_consent(self, reason: str) -> None:
        """Handles an explicit refusal with a reason.

        Args:
            reason: A short explanation of why the caller declined.
        """
        if not self.done():
            self.complete(RecordingConsentResult(consent=False, declined_reason=reason))
