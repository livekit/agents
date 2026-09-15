from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

from .events import AMDCategory, AMDCompletedEvent, AMDPredictionEvent

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
_AMDEvent = AMDPredictionEvent | AMDCompletedEvent
_HISTORY_LIMIT = 20


class _Lifecycle(Enum):
    NEW = auto()
    WAITING = auto()
    LISTENING = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class _Transcript:
    text: str
    source: Literal["session", "amd"] | None
    alternative: str = ""

    def context(self) -> dict[str, object]:
        data: dict[str, object] = {"transcript": self.text, "transcript_source": self.source}
        if self.alternative and self.alternative != self.text:
            data["alternative_transcript"] = self.alternative
        return data


_FallbackReason = Literal["reused", "superseded", "inference_timeout", "inference_error"]


@dataclass(frozen=True)
class _Prediction:
    category: AMDCategory
    fallback: _FallbackReason | None = None
    inference_duration: float | None = None


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
    prediction: _Prediction | None = None
    inference_deadline: float | None = None
    release_deadline: float | None = None
    timed_out: bool = False
    updated_turn_ids: set[int] = field(default_factory=set)

    def context(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            **self.transcript.context(),
            "dtmf_digits": self.dtmf_digits,
        }


class _AMDFSM:
    """AMD state and policy, with explicit timestamps and no async resources.

    A turn can release a fallback reply while its inference still runs.
    Its decision and pending prediction therefore have separate lifetimes.
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
        self._completion_reason = ""
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
        deadlines = [self._hard_deadline, self._idle_deadline]
        for turn in self._turns.values():
            deadlines.extend((turn.inference_deadline, turn.release_deadline))
        return min((at for at in deadlines if at is not None), default=None)

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
            turn.release_deadline = None

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
                events.extend(self._release(turn, now))
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
        self._updated_turn_ids.intersection_update(
            range(max(1, self.turn_id - _HISTORY_LIMIT + 1), self.turn_id + 1)
        )
        self._pending_dtmf_digits = ""
        return turn.turn_id

    def transcript_updated(self, turn_id: int, transcript: _Transcript) -> None:
        if self.finished or turn_id <= self.turn_id - _HISTORY_LIMIT:
            return
        turn = self._turns[turn_id]
        if turn.transcript != transcript:
            turn.transcript = transcript
            self._updated_turn_ids.add(turn_id)

    def transcript_ready(
        self, turn_id: int, transcript: _Transcript, now: float
    ) -> tuple[dict[str, object] | None, list[_AMDEvent]]:
        if self.finished:
            return None, []
        turn = self._turns[turn_id]
        turn.inference_text = transcript.text
        if turn_id < self._last_inference_turn_id:
            self._fallback(turn, "superseded", now)
            return None, []
        # ponytail: history is a window into the turn store, not a second collection.
        history_ids = set(range(max(1, self.turn_id - _HISTORY_LIMIT + 1), turn_id))
        if not turn.inference_text and not self._updated_turn_ids & history_ids:
            if self._pending_turn is not None and (
                self._pending_turn.decision is None or self._pending_turn.prediction is not None
            ):
                self._reused_turns.append(turn)
                self._pending_turn.release_epoch = turn.release_epoch
                self._pending_turn.silence_started_at = turn.silence_started_at
                return None, self._release(self._pending_turn, now)
            self._supersede(turn_id, now)
            return None, self._fallback(turn, "reused", now)
        self._supersede(turn_id, now)
        if self._pending_turn is not None:
            self._updated_turn_ids.update(self._pending_turn.updated_turn_ids)
            self._fallback(self._pending_turn, "superseded", now)
            self._flush_reused(now, reason="superseded")
        turn.updated_turn_ids = self._updated_turn_ids & history_ids
        self._updated_turn_ids.difference_update(turn.updated_turn_ids | {turn_id})
        self._last_inference_turn_id = turn_id
        self._pending_turn = turn
        turn.inference_deadline = turn.committed_at + self._inference_timeout
        return {
            "stage": self._category.value,
            "allowed_next_categories": sorted(ALLOWED[self._category]),
            "earlier_turns": [self._turns[index].context() for index in sorted(history_ids)],
            "updated_turn_ids": sorted(turn.updated_turn_ids),
            "speech_duration": turn.speech_duration,
            "turn_id": turn_id,
            "transcript": transcript.text,
            "transcript_source": transcript.source,
            "dtmf_digits": turn.dtmf_digits,
        }, []

    def prediction_received(
        self, turn_id: int, category: AMDCategory, now: float, inference_duration: float
    ) -> list[_AMDEvent]:
        if self.finished or self._pending_turn is None or self._pending_turn.turn_id != turn_id:
            return []
        turn = self._pending_turn
        category = self._category if category == AMDCategory.UNCERTAIN else category
        if category not in ALLOWED[self._category]:
            raise ValueError("invalid amd stage transition")
        turn.inference_deadline = None
        self._inference_timeouts = 0
        turn.prediction = _Prediction(category, inference_duration=inference_duration)
        return self._release(turn, now)

    def on_prediction(self, event: AMDPredictionEvent) -> _AMDEvent:
        """Apply a released model prediction and run its category handler."""
        self._latest = event
        self._category = event.category
        self._previous_turn = event.prev_turn_category
        self._previous_stage = event.prev_stage_category
        self._pending_turn = None
        if event.state_changed:
            self._idle_deadline = None
        if self._category != AMDCategory.UNCERTAIN:
            self._uncertain_turns = 0
        handlers: dict[AMDCategory, Callable[[AMDPredictionEvent], _AMDEvent]] = {
            AMDCategory.HUMAN: self.on_human_turn,
            AMDCategory.MACHINE_SCREENING: self.on_screening_turn,
            AMDCategory.MACHINE_IVR: self.on_ivr_turn,
            AMDCategory.MACHINE_VM: self.on_vm_turn,
            AMDCategory.UNCERTAIN: self.on_uncertain_turn,
            AMDCategory.MACHINE_UNAVAILABLE: self.on_unavailable_turn,
        }
        return handlers[event.category](event)

    def on_human_turn(self, event: AMDPredictionEvent) -> AMDCompletedEvent:
        return self.finish("finished")

    def on_screening_turn(self, event: AMDPredictionEvent) -> AMDPredictionEvent:
        return event

    def on_ivr_turn(self, event: AMDPredictionEvent) -> AMDPredictionEvent:
        return event

    def on_vm_turn(self, event: AMDPredictionEvent) -> AMDPredictionEvent:
        if event.state_changed:
            self._voicemail_reply_reserved = False
        return event

    def on_uncertain_turn(self, event: AMDPredictionEvent) -> _AMDEvent:
        self._uncertain_turns += 1
        if self._uncertain_turns >= self._max_uncertain_turns:
            return self.finish("max_uncertain_turns")
        return event

    def on_unavailable_turn(self, event: AMDPredictionEvent) -> AMDCompletedEvent:
        return self.finish("finished")

    def inference_failed(self, turn_id: int, now: float) -> list[_AMDEvent]:
        if self.finished or self._pending_turn is None or self._pending_turn.turn_id != turn_id:
            return []
        events = self._fallback(self._pending_turn, "inference_error", now)
        self._pending_turn = None
        return [*events, *self._flush_reused(now)]

    def tick(self, now: float) -> list[_AMDEvent]:
        """Apply the next due deadline, even when the timer wakes late."""
        due = self.next_deadline
        if due is None or now < due:
            return []
        if due == self._hard_deadline:
            return [self.finish("timeout")]
        if due == self._idle_deadline:
            return [self.finish("idle_timeout")]
        events: list[_AMDEvent] = []
        for turn in self._turns.values():
            if turn.inference_deadline == due:
                turn.inference_deadline = None
                turn.timed_out = True
                self._inference_timeouts += 1
                events.extend(self._fallback(turn, "inference_timeout", now))
                events.extend(self._flush_reused(now))
            if turn.release_deadline == due:
                events.extend(self._release(turn, now))
        return events

    def update_idle(self, now: float, *, session_busy: bool) -> None:
        if (
            not self.started
            or self._speaking_since is not None
            or session_busy
            or any(
                turn.decision is None or turn.prediction is not None
                for turn in self._turns.values()
            )
        ):
            self._idle_deadline = None
        elif self._idle_deadline is None:
            timeout = (
                self._voicemail_idle_timeout
                if self._category == AMDCategory.MACHINE_VM
                else self._idle_timeout
            )
            self._idle_deadline = now + timeout

    def authorize_reply(self, turn_id: int | None) -> AMDCategory | None:
        """Reserve a reply and return its instruction category, or None to skip it."""
        if turn_id not in self._turns:
            return (
                None
                if self.finished and self._category == AMDCategory.MACHINE_UNAVAILABLE
                else AMDCategory.UNCERTAIN
            )
        if turn_id != self.turn_id:
            return None
        if self.finished:
            if self._category == AMDCategory.MACHINE_UNAVAILABLE:
                return None
            return (
                AMDCategory.HUMAN if self._category == AMDCategory.HUMAN else AMDCategory.UNCERTAIN
            )
        if self._turns[turn_id].decision is None:
            return None
        if self._category == AMDCategory.MACHINE_VM:
            if self._voicemail_reply_reserved:
                return None
            self._voicemail_reply_reserved = True
        return self._category

    def voicemail_played(self) -> None:
        self._voicemail_message_played = True

    def finish(self, reason: str) -> AMDCompletedEvent:
        if self.finished:
            return self.completion()
        self._lifecycle = _Lifecycle.FINISHED
        self._completion_reason = reason
        self._hard_deadline = self._idle_deadline = None
        self._pending_dtmf_digits = ""
        for turn in self._turns.values():
            turn.prediction = None
            turn.inference_deadline = turn.release_deadline = None
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

    def _supersede(self, turn_id: int, now: float) -> None:
        for previous in self._turns.values():
            if previous.turn_id < turn_id and previous.prediction is not None:
                self._fallback(previous, "superseded", now)

    def _flush_reused(self, now: float, *, reason: _FallbackReason = "reused") -> list[_AMDEvent]:
        turns, self._reused_turns = self._reused_turns, []
        events: list[_AMDEvent] = []
        for turn in turns:
            events.extend(self._fallback(turn, reason, now))
        return events

    def _fallback(self, turn: _Turn, reason: _FallbackReason, now: float) -> list[_AMDEvent]:
        turn.inference_deadline = turn.release_deadline = None
        turn.prediction = None
        if turn.decision is not None:
            return []
        prediction = _Prediction(self._category, fallback=reason)
        if reason == "superseded":
            turn.decision = self._event(turn, prediction, now)
            return []
        turn.prediction = prediction
        return self._release(turn, now)

    def _release(self, turn: _Turn, now: float) -> list[_AMDEvent]:
        turn.release_deadline = None
        prediction = turn.prediction
        if self.finished or prediction is None:
            return []
        if (
            prediction.category not in {AMDCategory.HUMAN, AMDCategory.UNCERTAIN}
            and self._machine_silence_threshold > 0
        ):
            self._idle_deadline = None
            if self._speaking_since is not None or turn.release_epoch != self._speech_epoch:
                return []
            release_at = turn.silence_started_at + self._machine_silence_threshold
            if now < release_at:
                turn.release_deadline = release_at
                return []
        turn.prediction = None
        event = self._event(turn, prediction, now)
        if turn.decision is None:
            turn.decision = event
        if prediction.fallback is not None:
            self._latest = event
            if prediction.fallback == "inference_timeout" and self._inference_timeouts >= 3:
                return [event.model_copy(), self.finish("inference_timeout")]
            return [event.model_copy()]
        result = self.on_prediction(event)
        if isinstance(result, AMDCompletedEvent):
            return [event.model_copy(), result]
        return [result.model_copy(), *self._flush_reused(now)]

    def _event(self, turn: _Turn, prediction: _Prediction, now: float) -> AMDPredictionEvent:
        return AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=prediction.category,
            reason=prediction.fallback or ("late_prediction" if turn.timed_out else "prediction"),
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
