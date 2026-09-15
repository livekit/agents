from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from typing import Literal

from pydantic import BaseModel

from .events import AMDCategory, AMDCompletedEvent, AMDPredictionEvent, AMDReason

ALLOWED = {
    AMDCategory.UNCERTAIN: frozenset(AMDCategory),
    AMDCategory.MACHINE_SCREENING: frozenset(
        {
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
        }
    ),
    AMDCategory.MACHINE_VM: frozenset(
        {
            AMDCategory.MACHINE_VM,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_UNAVAILABLE,
        }
    ),
    AMDCategory.MACHINE_IVR: frozenset(
        {
            AMDCategory.MACHINE_IVR,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
        }
    ),
}
TERMINAL = frozenset({AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE})
_AMDEvent = AMDPredictionEvent | AMDCompletedEvent
_Deadline = tuple[float, Callable[[float], list[_AMDEvent]]]
_Source = Literal["session", "amd"]
_HISTORY_LIMIT = 20
_MAX_INFERENCE_TIMEOUTS = 3


class _Lifecycle(Enum):
    NEW = auto()
    WAITING = auto()
    LISTENING = auto()
    FINISHED = auto()


class _Phase(Enum):
    """Prediction pipeline of one turn. Reply authorization lives in ``_Turn.decision``."""

    IDLE = auto()
    """No model result is pending."""
    INFERRING = auto()
    """The classifier is running. ``deadline`` is the inference timeout."""
    HOLDING = auto()
    """A machine prediction waits for participant silence. ``deadline`` is the release time,
    or None while the participant is still speaking."""


@dataclass(frozen=True)
class _Transcript:
    text: str
    source: _Source | None
    alternative: str = ""


@dataclass(frozen=True)
class _Prediction:
    category: AMDCategory
    fallback: AMDReason | None = None
    inference_duration: float | None = None


@dataclass(frozen=True)
class ReplyDecision:
    allow: bool
    instructions_for: AMDCategory | None = None


_SKIP = ReplyDecision(allow=False)
_PLAIN = ReplyDecision(allow=True)


class TurnContext(BaseModel):
    turn_id: int
    transcript: str
    transcript_source: _Source | None
    dtmf_digits: str
    alternative_transcript: str | None = None


class ClassifyRequest(BaseModel):
    """Model-facing payload for one classification. Serialize with ``exclude_none``."""

    stage: AMDCategory
    allowed_next_categories: list[AMDCategory]
    earlier_turns: list[TurnContext]
    updated_turn_ids: list[int]
    speech_duration: float
    turn_id: int
    transcript: str
    transcript_source: _Source | None
    dtmf_digits: str


@dataclass
class _Turn:
    turn_id: int
    committed_at: float
    transcript: _Transcript
    speech_duration: float
    release_epoch: int
    silence_started_at: float
    dtmf_digits: str
    inference_text: str = ""
    decision: AMDPredictionEvent | None = None
    phase: _Phase = _Phase.IDLE
    deadline: float | None = None
    prediction: _Prediction | None = None
    timed_out: bool = False
    updated_turn_ids: set[int] = field(default_factory=set)

    def context(self) -> TurnContext:
        alternative = self.transcript.alternative
        return TurnContext(
            turn_id=self.turn_id,
            transcript=self.transcript.text,
            transcript_source=self.transcript.source,
            dtmf_digits=self.dtmf_digits,
            alternative_transcript=alternative
            if alternative and alternative != self.transcript.text
            else None,
        )

    def settle(self) -> None:
        self.phase = _Phase.IDLE
        self.deadline = None
        self.prediction = None


class _AMDFSM:
    """AMD state and policy, with explicit timestamps and no async resources.

    A turn can release a fallback reply while its inference still runs.
    Its decision and prediction pipeline therefore have separate lifetimes.
    """

    def __init__(
        self,
        *,
        idle_timeout: float,
        voicemail_idle_timeout: float,
        timeout: float,
        inference_timeout: float,
        machine_silence_threshold: float,
        max_uncertain_turns: int,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._voicemail_idle_timeout = voicemail_idle_timeout
        self._timeout = timeout
        self._inference_timeout = inference_timeout
        self._machine_silence_threshold = machine_silence_threshold
        self._max_uncertain_turns = max_uncertain_turns
        self._lifecycle = _Lifecycle.NEW
        self._completion_reason = AMDReason.CANCELLED
        self._category = AMDCategory.UNCERTAIN
        self._previous_turn: AMDCategory | None = None
        self._previous_stage: AMDCategory | None = None
        self._latest: AMDPredictionEvent | None = None
        self._voicemail_reply_reserved = False
        self._voicemail_message_played = False
        self._uncertain_turns = 0
        self._inference_timeouts = 0
        self._speaking_since: float | None = None
        self._speech_ended_at: float | None = None
        self._speech_duration = 0.0
        self._speech_epoch = 0
        self._turns: dict[int, _Turn] = {}
        self._pending_turn: _Turn | None = None
        self._last_inference_turn_id = 0
        self._reused_turns: list[_Turn] = []
        self._updated_turn_ids: set[int] = set()
        self._pending_dtmf_digits = ""
        self._hard_deadline: float | None = None
        self._idle_deadline: float | None = None

    @property
    def entered(self) -> bool:
        return self._lifecycle is not _Lifecycle.NEW

    @property
    def enabled(self) -> bool:
        return self._lifecycle in {_Lifecycle.WAITING, _Lifecycle.LISTENING}

    @property
    def started(self) -> bool:
        return self._lifecycle is _Lifecycle.LISTENING

    @property
    def finished(self) -> bool:
        return self._lifecycle is _Lifecycle.FINISHED

    @property
    def category(self) -> AMDCategory:
        return self._category

    @property
    def turn_id(self) -> int:
        return len(self._turns)

    @property
    def voicemail_message_played(self) -> bool:
        return self._voicemail_message_played

    @property
    def next_deadline(self) -> float | None:
        return min((at for at, _ in self._deadlines()), default=None)

    def has_turn(self, turn_id: int | None) -> bool:
        return turn_id in self._turns

    def decision(self, turn_id: int) -> AMDPredictionEvent | None:
        decision = self._turns[turn_id].decision
        return decision.model_copy() if decision is not None else None

    def enter(self) -> None:
        if self.entered:
            raise RuntimeError("use a new AMD instance for each run")
        self._lifecycle = _Lifecycle.WAITING

    def start(self, now: float) -> None:
        if self._lifecycle is _Lifecycle.WAITING:
            self._lifecycle = _Lifecycle.LISTENING
            self._hard_deadline = now + self._timeout

    def dtmf_sent(self, digits: str) -> None:
        if not self.enabled:
            return
        if not digits or any(digit not in "0123456789*#ABCD" for digit in digits):
            raise ValueError("digits must contain only 0-9, *, #, or A-D")
        self._pending_dtmf_digits += digits

    def speech_started(self, now: float) -> None:
        if not self.started:
            return
        self._speech_epoch += 1
        self._speaking_since = now
        self._speech_ended_at = None
        self._idle_deadline = None
        for turn in self._turns.values():
            if turn.phase is _Phase.HOLDING:
                turn.deadline = None

    def speech_ended(self, now: float, silence_duration: float) -> list[_AMDEvent]:
        if not self.started:
            return []
        self._speech_ended_at = now - max(0, silence_duration)
        if self._speaking_since is not None:
            self._speech_duration += max(0, self._speech_ended_at - self._speaking_since)
        self._speaking_since = None
        events: list[_AMDEvent] = []
        for turn in self._turns.values():
            if turn.release_epoch == self._speech_epoch:
                turn.silence_started_at = self._speech_ended_at
                events.extend(self._resume(turn, now))
        return events

    def commit_turn(self, transcript: _Transcript, now: float, eot_delay: float) -> int:
        if not self.started:
            raise RuntimeError("AMD must be listening before committing a turn")
        self._idle_deadline = None
        speech_duration, self._speech_duration = self._speech_duration, 0
        if self._speaking_since is not None:
            speech_duration += now - self._speaking_since
            self._speaking_since = now
        turn = _Turn(
            turn_id=self.turn_id + 1,
            committed_at=now,
            transcript=transcript,
            speech_duration=speech_duration,
            release_epoch=self._speech_epoch,
            silence_started_at=self._speech_ended_at
            if self._speech_ended_at is not None
            else now - max(0, eot_delay),
            dtmf_digits=self._pending_dtmf_digits,
            inference_text=transcript.text,
        )
        self._turns[turn.turn_id] = turn
        self._updated_turn_ids.intersection_update(self._history_ids(self.turn_id + 1))
        self._pending_dtmf_digits = ""
        return turn.turn_id

    def transcript_updated(self, turn_id: int, transcript: _Transcript) -> None:
        if self.finished or turn_id not in self._history_ids(self.turn_id + 1):
            return
        turn = self._turns[turn_id]
        if turn.transcript != transcript:
            turn.transcript = transcript
            self._updated_turn_ids.add(turn_id)

    def transcript_ready(
        self, turn_id: int, transcript: _Transcript, now: float
    ) -> tuple[ClassifyRequest | None, list[_AMDEvent]]:
        if self.finished:
            return None, []
        turn = self._turns[turn_id]
        turn.inference_text = transcript.text
        if turn_id < self._last_inference_turn_id:
            self._fallback(turn, AMDReason.SUPERSEDED, now)
            return None, []
        history_ids = set(self._history_ids(turn_id))
        if not turn.inference_text and not self._updated_turn_ids & history_ids:
            return None, self._reuse_pending(turn, now)
        self._supersede_pending(turn_id, now)
        return self._new_request(turn, transcript, history_ids), []

    def prediction_received(
        self, turn_id: int, category: AMDCategory, now: float, inference_duration: float
    ) -> list[_AMDEvent]:
        if self.finished or self._pending_turn is None or self._pending_turn.turn_id != turn_id:
            return []
        turn = self._pending_turn
        category = self._category if category == AMDCategory.UNCERTAIN else category
        if category not in ALLOWED[self._category]:
            raise ValueError("invalid amd stage transition")
        self._inference_timeouts = 0
        return self._release(
            turn, _Prediction(category, inference_duration=inference_duration), now
        )

    def inference_failed(self, turn_id: int, now: float) -> list[_AMDEvent]:
        if self.finished or self._pending_turn is None or self._pending_turn.turn_id != turn_id:
            return []
        events = self._fallback(self._pending_turn, AMDReason.INFERENCE_ERROR, now)
        self._pending_turn = None
        return [*events, *self._flush_reused(now)]

    def tick(self, now: float) -> list[_AMDEvent]:
        """Apply every due deadline in order, even when the timer wakes late."""
        events: list[_AMDEvent] = []
        for at, apply in sorted(self._deadlines(), key=lambda deadline: deadline[0]):
            if at > now or self.finished:
                break
            events.extend(apply(now))
        return events

    def update_idle(self, now: float, *, session_busy: bool) -> None:
        unsettled = any(
            turn.decision is None or turn.phase is _Phase.HOLDING for turn in self._turns.values()
        )
        if not self.started or self._speaking_since is not None or session_busy or unsettled:
            self._idle_deadline = None
        elif self._idle_deadline is None:
            timeout = (
                self._voicemail_idle_timeout
                if self._category == AMDCategory.MACHINE_VM
                else self._idle_timeout
            )
            self._idle_deadline = now + timeout

    def authorize_reply(self, turn_id: int | None) -> ReplyDecision:
        """Reserve a reply for the turn and select its stage instructions."""
        if self.finished and self._category is AMDCategory.MACHINE_UNAVAILABLE:
            return _SKIP
        if turn_id not in self._turns:
            return _PLAIN
        if turn_id != self.turn_id or self._turns[turn_id].decision is None:
            return _SKIP
        if self.finished:
            human = self._category is AMDCategory.HUMAN
            return ReplyDecision(allow=True, instructions_for=AMDCategory.HUMAN if human else None)
        if self._category is AMDCategory.MACHINE_VM:
            if self._voicemail_reply_reserved:
                return _SKIP
            self._voicemail_reply_reserved = True
        if self._category is AMDCategory.UNCERTAIN:
            return _PLAIN
        return ReplyDecision(allow=True, instructions_for=self._category)

    def voicemail_played(self) -> None:
        self._voicemail_message_played = True

    def finish(self, reason: AMDReason) -> AMDCompletedEvent:
        if self.finished:
            return self.completion()
        self._lifecycle = _Lifecycle.FINISHED
        self._completion_reason = reason
        self._hard_deadline = self._idle_deadline = None
        self._pending_dtmf_digits = ""
        for turn in self._turns.values():
            turn.settle()
            if turn.decision is None:
                turn.decision = self._latest or AMDPredictionEvent(
                    turn_id=turn.turn_id,
                    category=self._category,
                    reason=reason,
                    transcript=turn.inference_text,
                    speech_duration=turn.speech_duration,
                    delay=0,
                )
        return self.completion()

    def completion(self) -> AMDCompletedEvent:
        if not self.finished:
            raise RuntimeError("AMD has not completed")
        return AMDCompletedEvent(
            category=self._category,
            reason=self._completion_reason,
            turn_id=self._latest.turn_id if self._latest else self.turn_id,
            transcript=self._latest.transcript if self._latest else "",
            prev_turn_category=self._previous_turn,
            prev_stage_category=self._previous_stage,
            voicemail_message_played=self._voicemail_message_played,
        )

    def _history_ids(self, before: int) -> range:
        return range(max(1, self.turn_id - _HISTORY_LIMIT + 1), before)

    def _deadlines(self) -> Iterator[_Deadline]:
        if self._hard_deadline is not None:
            yield self._hard_deadline, partial(self._expire, AMDReason.TIMEOUT)
        if self._idle_deadline is not None:
            yield self._idle_deadline, partial(self._expire, AMDReason.IDLE_TIMEOUT)
        for turn in self._turns.values():
            if turn.deadline is None:
                continue
            if turn.phase is _Phase.INFERRING:
                yield turn.deadline, partial(self._inference_timed_out, turn)
            elif turn.phase is _Phase.HOLDING:
                yield turn.deadline, partial(self._resume, turn)

    def _expire(self, reason: AMDReason, now: float) -> list[_AMDEvent]:
        return [self.finish(reason)]

    def _inference_timed_out(self, turn: _Turn, now: float) -> list[_AMDEvent]:
        turn.timed_out = True
        self._inference_timeouts += 1
        events = self._fallback(turn, AMDReason.INFERENCE_TIMEOUT, now)
        return [*events, *self._flush_reused(now)]

    def _reuse_pending(self, turn: _Turn, now: float) -> list[_AMDEvent]:
        """An empty turn re-anchors the pending turn's silence instead of a new request."""
        pending = self._pending_turn
        if pending is not None and (pending.decision is None or pending.phase is _Phase.HOLDING):
            self._reused_turns.append(turn)
            pending.release_epoch = turn.release_epoch
            pending.silence_started_at = turn.silence_started_at
            return self._resume(pending, now)
        self._supersede(turn.turn_id, now)
        return self._fallback(turn, AMDReason.REUSED, now)

    def _supersede_pending(self, turn_id: int, now: float) -> None:
        """Resolve every older turn's pending work before a new request starts."""
        self._supersede(turn_id, now)
        if self._pending_turn is not None:
            self._updated_turn_ids.update(self._pending_turn.updated_turn_ids)
            self._fallback(self._pending_turn, AMDReason.SUPERSEDED, now)
            self._flush_reused(now, reason=AMDReason.SUPERSEDED)

    def _new_request(
        self, turn: _Turn, transcript: _Transcript, history_ids: set[int]
    ) -> ClassifyRequest:
        turn.updated_turn_ids = self._updated_turn_ids & history_ids
        self._updated_turn_ids.difference_update(turn.updated_turn_ids | {turn.turn_id})
        self._last_inference_turn_id = turn.turn_id
        self._pending_turn = turn
        turn.phase = _Phase.INFERRING
        turn.deadline = turn.committed_at + self._inference_timeout
        return ClassifyRequest(
            stage=self._category,
            allowed_next_categories=sorted(ALLOWED[self._category]),
            earlier_turns=[self._turns[index].context() for index in sorted(history_ids)],
            updated_turn_ids=sorted(turn.updated_turn_ids),
            speech_duration=turn.speech_duration,
            turn_id=turn.turn_id,
            transcript=transcript.text,
            transcript_source=transcript.source,
            dtmf_digits=turn.dtmf_digits,
        )

    def _supersede(self, turn_id: int, now: float) -> None:
        for previous in self._turns.values():
            if previous.turn_id < turn_id and previous.phase is _Phase.HOLDING:
                self._fallback(previous, AMDReason.SUPERSEDED, now)

    def _flush_reused(self, now: float, *, reason: AMDReason = AMDReason.REUSED) -> list[_AMDEvent]:
        turns, self._reused_turns = self._reused_turns, []
        events: list[_AMDEvent] = []
        for turn in turns:
            events.extend(self._fallback(turn, reason, now))
        return events

    def _fallback(self, turn: _Turn, reason: AMDReason, now: float) -> list[_AMDEvent]:
        turn.settle()
        if turn.decision is not None:
            return []
        prediction = _Prediction(self._category, fallback=reason)
        if reason is AMDReason.SUPERSEDED:
            turn.decision = self._event(turn, prediction, now)
            return []
        return self._release(turn, prediction, now)

    def _resume(self, turn: _Turn, now: float) -> list[_AMDEvent]:
        if turn.phase is not _Phase.HOLDING:
            return []
        assert turn.prediction is not None
        return self._release(turn, turn.prediction, now)

    def _release(self, turn: _Turn, prediction: _Prediction, now: float) -> list[_AMDEvent]:
        """Publish a prediction, or hold it until the participant has been silent long enough."""
        turn.settle()
        if self.finished:
            return []
        if prediction.category not in {AMDCategory.HUMAN, AMDCategory.UNCERTAIN} and (
            self._machine_silence_threshold > 0
        ):
            self._idle_deadline = None
            speaking = self._speaking_since is not None or turn.release_epoch != self._speech_epoch
            release_at = turn.silence_started_at + self._machine_silence_threshold
            if speaking or now < release_at:
                turn.phase = _Phase.HOLDING
                turn.prediction = prediction
                turn.deadline = None if speaking else release_at
                return []
        event = self._event(turn, prediction, now)
        if turn.decision is None:
            turn.decision = event
        self._latest = event
        if prediction.fallback is not None:
            if (
                prediction.fallback is AMDReason.INFERENCE_TIMEOUT
                and self._inference_timeouts >= _MAX_INFERENCE_TIMEOUTS
            ):
                return [event.model_copy(), self.finish(AMDReason.INFERENCE_TIMEOUT)]
            return [event.model_copy()]
        result = self._apply(event)
        if isinstance(result, AMDCompletedEvent):
            return [event.model_copy(), result]
        return [event.model_copy(), *self._flush_reused(now)]

    def _apply(self, event: AMDPredictionEvent) -> _AMDEvent:
        """Move the stage to a released model prediction."""
        self._category = event.category
        self._previous_turn = event.prev_turn_category
        self._previous_stage = event.prev_stage_category
        self._pending_turn = None
        if event.state_changed:
            self._idle_deadline = None
            self._voicemail_reply_reserved = False
        self._uncertain_turns = (
            self._uncertain_turns + 1 if self._category is AMDCategory.UNCERTAIN else 0
        )
        if self._category in TERMINAL:
            return self.finish(AMDReason.FINISHED)
        if self._uncertain_turns >= self._max_uncertain_turns:
            return self.finish(AMDReason.MAX_UNCERTAIN_TURNS)
        return event

    def _event(self, turn: _Turn, prediction: _Prediction, now: float) -> AMDPredictionEvent:
        if prediction.fallback is not None:
            reason = prediction.fallback
        elif turn.timed_out:
            reason = AMDReason.LATE_PREDICTION
        else:
            reason = AMDReason.PREDICTION
        return AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=prediction.category,
            reason=reason,
            transcript=turn.inference_text,
            speech_duration=turn.speech_duration,
            delay=now - turn.committed_at,
            inference_duration=prediction.inference_duration,
            prev_turn_category=self._latest.category if self._latest else None,
            prev_stage_category=self._category
            if prediction.fallback is None and prediction.category != self._category
            else self._previous_stage,
            voicemail_message_played=self._voicemail_message_played,
        )
