"""AMD turn bookkeeping and independent classifier history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ... import llm
from .events import AMDCategory, AMDPredictionEvent

AMDTranscriptSource = Literal["session", "amd"]
_HISTORY_LIMIT = 20


@dataclass(frozen=True)
class AMDTranscript:
    transcript: str
    source: AMDTranscriptSource | None


@dataclass
class Turn:
    turn_id: int
    committed_at: float
    transcript: AMDTranscript
    speech_duration: float
    prediction: AMDPredictionEvent | None = None
    inference_duration: float | None = None


@dataclass(frozen=True)
class AMDRequest:
    """Conversation snapshot and constraints for one classification."""

    stage: AMDCategory
    allowed_next_categories: list[AMDCategory]
    chat_ctx: llm.ChatContext
    speech_duration: float


class AMDChatContext(llm.ChatContext):
    """A thin wrapper to track only user transcript and DTMF events."""

    def add_transcript(self, turn: Turn) -> None:
        self.add_message(
            role="user",
            content=turn.transcript.transcript,
            extra={"turn_id": turn.turn_id, "transcript_source": turn.transcript.source},
        )

    def add_tool_result(self, call: llm.FunctionCall, output: llm.FunctionCallOutput) -> None:
        if (
            call.name != "send_dtmf_events"
            or output.is_error
            or not output.output
            or self.get_by_id(call.id) is not None
        ):
            return
        self.items.extend([call.model_copy(), output.model_copy()])

    def create_request(
        self, turn: Turn, *, stage: AMDCategory, allowed: list[AMDCategory]
    ) -> AMDRequest:
        return AMDRequest(
            stage=stage,
            allowed_next_categories=allowed,
            chat_ctx=self.truncate(max_items=_HISTORY_LIMIT).copy(),
            speech_duration=turn.speech_duration,
        )
