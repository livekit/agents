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


def update_transcript(chat_ctx: llm.ChatContext, turn: Turn) -> None:
    """Replace a turn's transcript message, as when its AMD final arrives late."""
    for item in chat_ctx.items:
        if item.type == "message" and item.extra.get("turn_id") == turn.turn_id:
            item.content = [turn.transcript.for_classifier()]
            return


@dataclass(frozen=True)
class AMDRequest:
    """Conversation snapshot and constraints for one classification."""

    stage: AMDCategory
    recommended_next_categories: list[AMDCategory]
    previous_prediction: AMDPredictionEvent | None
    chat_ctx: llm.ChatContext
    speech_duration: float
    likely_human: bool = False
    """The previous prediction is likely human, counted as uncertain."""


class AMDChatContext(llm.ChatContext):
    """A thin wrapper to track only user transcript and DTMF events."""

    def add_transcript(self, turn: Turn) -> None:
        self.add_message(
            role="user",
            content=turn.transcript.for_classifier(),
            extra={"turn_id": turn.turn_id},
        )

    def add_agent_reply(self, started_at: float, text: str, *, words: bool = False) -> None:
        """Mark that the agent spoke, at the time its speech started.

        The session adds a reply when its speech ends, so a participant turn committed
        during that speech can arrive first. Inserting by start time keeps speaking order.
        With `words`, keep what it said: after a likely human turn, a reply to the agent
        shows a person. The prompt still says the words are hidden; rewording it moved
        machine goodbyes between categories.
        """
        content = text if words else AGENT_REPLY
        self.insert(llm.ChatMessage(role="assistant", content=[content], created_at=started_at))

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
        likely_human: bool = False,
    ) -> AMDRequest:
        return AMDRequest(
            stage=stage,
            recommended_next_categories=recommended,
            previous_prediction=previous_prediction.model_copy() if previous_prediction else None,
            chat_ctx=self.truncate(max_items=_HISTORY_LIMIT).copy(),
            speech_duration=turn.speech_duration,
            likely_human=likely_human,
        )
