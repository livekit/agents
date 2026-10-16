"""AMD turn bookkeeping and independent classifier history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ... import llm
from .events import AMDCategory, AMDPredictionEvent

AMDTranscriptSource = Literal["session", "amd"]
_HISTORY_LIMIT = 20
AGENT_REPLY = "(the agent spoke)"


@dataclass(frozen=True)
class AMDTranscript:
    """A turn's transcript from each STT; None for an STT that is not in use."""

    session: str | None
    amd: str | None = None

    @property
    def transcript(self) -> str:
        """The one transcript for events and the Agent: the session's, else AMD's."""
        return self.session or self.amd or ""

    def for_classifier(self) -> str:
        """Both transcripts when AMD has its own STT, so neither has to win a race.

        Encoded as JSON: as bare labeled lines, the classifier stopped reading a
        lone "Hi." as only a greeting word.
        """
        if self.session is None or self.amd is None:
            return self.transcript
        return json.dumps({"transcript_1": self.session, "transcript_2": self.amd})


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
    recommended_next_categories: list[AMDCategory]
    previous_prediction: AMDPredictionEvent | None
    chat_ctx: llm.ChatContext
    speech_duration: float


class AMDChatContext(llm.ChatContext):
    """A thin wrapper to track only user transcript and DTMF events."""

    def add_transcript(self, turn: Turn) -> None:
        self.add_message(
            role="user",
            content=turn.transcript.for_classifier(),
            extra={"turn_id": turn.turn_id},
        )

    def add_agent_reply(self) -> None:
        # that the agent spoke, never its words
        self.add_message(role="assistant", content=AGENT_REPLY)

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
        self,
        turn: Turn,
        *,
        stage: AMDCategory,
        recommended: list[AMDCategory],
        previous_prediction: AMDPredictionEvent | None,
    ) -> AMDRequest:
        return AMDRequest(
            stage=stage,
            recommended_next_categories=recommended,
            previous_prediction=previous_prediction.model_copy() if previous_prediction else None,
            chat_ctx=self.truncate(max_items=_HISTORY_LIMIT).copy(),
            speech_duration=turn.speech_duration,
        )
