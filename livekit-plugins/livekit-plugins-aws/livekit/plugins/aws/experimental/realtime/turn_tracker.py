from __future__ import annotations

import datetime
import enum
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import llm

from ...log import logger

# Nova Sonic's barge-in detection signal (raw content without newline)
BARGE_IN_CONTENT = '{ "interrupted" : true }'


class _Phase(enum.Enum):
    IDLE = 0  # waiting for the USER to begin speaking
    USER_SPEAKING = 1  # still receiving USER text+audio blocks
    USER_FINISHED = 2  # first ASSISTANT speculative block observed
    ASSISTANT_RESPONDING = 3  # ASSISTANT audio/text streaming
    DONE = 4  # assistant audio ended (END_TURN) or barge-in (INTERRUPTED)


# note: b/c user ASR text is transcribed server-side, a single turn constitutes
# both the user and agent's speech
@dataclass
class _Turn:
    turn_id: int
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    transcript: list[str] = field(default_factory=list)

    phase: _Phase = _Phase.IDLE
    ev_input_started: bool = False
    ev_input_stopped: bool = False
    ev_trans_completed: bool = False
    ev_generation_sent: bool = False

    def add_partial_text(self, text: str) -> None:
        self.transcript.append(text)

    @property
    def curr_transcript(self) -> str:
        return " ".join(self.transcript)


class _TurnTracker:
    def __init__(
        self,
        emit_fn: Callable[[str, Any], None],
        emit_generation_fn: Callable[[], None],
    ):
        self._emit = emit_fn
        self._turn_idx = 0
        self._curr_turn: _Turn | None = None
        self._emit_generation_fn = emit_generation_fn

    # --------------------------------------------------------
    #  PUBLIC ENTRY POINT
    # --------------------------------------------------------
    def feed(self, event: dict) -> None:
        turn = self._ensure_turn()
        kind = _classify(event)

        if kind == "USER_TEXT_PARTIAL":
            turn.add_partial_text(event["event"]["textOutput"]["content"])
            self._maybe_emit_input_started(turn)
            self._emit_transcript_updated(turn)
            # note: cannot invoke self._maybe_input_stopped() here
            # b/c there is no way to know if the user is done speaking

        # will always be correlated b/c generate_reply() is a stub
        # user ASR text ends when agent's ASR speculative text begins
        # corresponds to beginning of agent's turn
        elif kind == "TOOL_OUTPUT_CONTENT_START" or kind == "ASSISTANT_SPEC_START":
            # must be a maybe methods b/c agent can chain multiple tool calls
            self._maybe_emit_input_stopped(turn)
            self._maybe_emit_transcript_completed(turn)
            self._maybe_emit_generation_created(turn)

        elif kind == "BARGE_IN":
            logger.debug(f"BARGE-IN DETECTED IN TURN TRACKER: {turn}")
            # start new turn immediately to make interruptions snappier
            self._emit("input_speech_started", llm.InputSpeechStartedEvent())
            turn.phase = _Phase.DONE

        elif kind == "ASSISTANT_AUDIO_END":
            if event["event"]["contentEnd"]["stopReason"] == "END_TURN":
                turn.phase = _Phase.DONE

        if turn.phase is _Phase.DONE:
            self._curr_turn = None

    def _ensure_turn(self) -> _Turn:
        if self._curr_turn is None:
            self._turn_idx += 1
            self._curr_turn = _Turn(turn_id=self._turn_idx)
        return self._curr_turn

    def _maybe_emit_input_started(self, turn: _Turn) -> None:
        if not turn.ev_input_started:
            turn.ev_input_started = True
            self._emit("input_speech_started", llm.InputSpeechStartedEvent())
            turn.phase = _Phase.USER_SPEAKING

    def _maybe_emit_input_stopped(self, turn: _Turn) -> None:
        if not turn.ev_input_stopped:
            turn.ev_input_stopped = True
            self._emit(
                "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
            )
            turn.phase = _Phase.USER_FINISHED

    def _emit_transcript_updated(self, turn: _Turn) -> None:
        self._emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=turn.input_id,
                transcript=turn.curr_transcript,
                is_final=False,
            ),
        )

    def _maybe_emit_transcript_completed(self, turn: _Turn) -> None:
        if not turn.ev_trans_completed:
            turn.ev_trans_completed = True
            self._emit(
                "input_audio_transcription_completed",
                # Q: does input_id need to match /w the _ResponseGeneration.input_id?
                llm.InputTranscriptionCompleted(
                    item_id=turn.input_id,
                    transcript=turn.curr_transcript,
                    is_final=True,
                ),
            )

    def _maybe_emit_generation_created(self, turn: _Turn) -> None:
        if not turn.ev_generation_sent:
            turn.ev_generation_sent = True
            logger.debug(
                f"[GEN] TurnTracker calling emit_generation_fn() for turn_id={turn.turn_id}"
            )
            self._emit_generation_fn()
            turn.phase = _Phase.ASSISTANT_RESPONDING
        else:
            logger.debug(f"[GEN] TurnTracker SKIPPED - already sent for turn_id={turn.turn_id}")


def _classify(ev: dict) -> str:
    e = ev.get("event", {})
    if "textOutput" in e and e["textOutput"]["role"] == "USER":
        return "USER_TEXT_PARTIAL"

    if "contentStart" in e and e["contentStart"]["type"] == "TOOL":
        return "TOOL_OUTPUT_CONTENT_START"

    if "contentStart" in e and e["contentStart"]["role"] == "ASSISTANT":
        add = e["contentStart"].get("additionalModelFields", "")
        if "SPECULATIVE" in add:
            return "ASSISTANT_SPEC_START"

    if "textOutput" in e and e["textOutput"]["content"] == BARGE_IN_CONTENT:
        return "BARGE_IN"

    # note: there cannot be any audio events for the user in the output event loop
    # therefore, we know that the audio event must be for the assistant
    if "contentEnd" in e and e["contentEnd"]["type"] == "AUDIO":
        return "ASSISTANT_AUDIO_END"

    return ""
