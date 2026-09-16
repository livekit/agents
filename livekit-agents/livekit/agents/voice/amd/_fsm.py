"""AMD turn and stage policy driven by explicit timestamps, with no asyncio resources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from pydantic import BaseModel

from .events import AMDCategory, AMDCompletedEvent, AMDPredictionEvent, AMDReason

_HISTORY_LIMIT = 19  # Earlier turns sent to the classifier with the current turn.

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
MACHINE = frozenset(
    {
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
        AMDCategory.MACHINE_UNAVAILABLE,
    }
)
AMDEvent = AMDPredictionEvent | AMDCompletedEvent
AMDTranscriptSource = Literal["session", "amd"]


class AMDLifecycle(Enum):
    """Lifecycle of an AMD run."""

    INITIALIZED = auto()
    PENDING = auto()
    ACTIVE = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class AMDTranscript:
    transcript: str
    source: AMDTranscriptSource | None


@dataclass(frozen=True)
class AMDPrediction:
    category: AMDCategory
    reason: AMDReason
    inference_duration: float | None = None

    @property
    def from_model(self) -> bool:
        return self.reason in {AMDReason.PREDICTION, AMDReason.LATE_PREDICTION}


@dataclass(frozen=True)
class AMDReplyDecision:
    allow: bool
    instructions_for: AMDCategory | None = None
    track_voicemail: bool = False


_SKIP = AMDReplyDecision(allow=False)
_ABSTAIN = AMDReplyDecision(allow=True)


class AMDTurnContext(BaseModel):
    turn_id: int
    transcript: str
    transcript_source: AMDTranscriptSource | None
    dtmf_digits: str


class AMDClassifyRequest(BaseModel):
    """Model-facing payload for one classification. Serialize with ``exclude_none``."""

    stage: AMDCategory
    allowed_next_categories: list[AMDCategory]
    current_turn: AMDTurnContext
    earlier_turns: list[AMDTurnContext]
    speech_duration: float


@dataclass(frozen=True)
class AMDMenuRequest:
    turn_id: int
    transcript: str


AMDEffect = AMDPredictionEvent | AMDCompletedEvent | AMDClassifyRequest | AMDMenuRequest
"""What a transition asks the driver to do: emit an event, classify a turn, or extract a menu."""
_Deadline = tuple[float, Callable[[float], list[AMDEffect]]]
"""An armed deadline and the transition to apply once it is due."""


@dataclass
class AMDTurnHandle:
    turn_id: int
    committed_at: float
    transcript: AMDTranscript
    speech_duration: float
    dtmf_digits: str
    reuses: int | None = None
    """Earlier turn whose outstanding prediction also settles this empty turn."""
    prediction: AMDPredictionEvent | None = None
    """The first prediction released for this turn."""

    def context(self) -> AMDTurnContext:
        return AMDTurnContext(
            turn_id=self.turn_id,
            transcript=self.transcript.transcript,
            transcript_source=self.transcript.source,
            dtmf_digits=self.dtmf_digits,
        )


@dataclass
class AMDInference:
    """The one classifier request in flight. A timed-out request may still return late."""

    turn: AMDTurnHandle
    deadline: float
    timed_out: bool = False


@dataclass
class AMDHold:
    """The one machine prediction waiting for user silence before release."""

    turn: AMDTurnHandle
    prediction: AMDPrediction
    release_at: float | None
    """Release time, or None while the user speaks."""


@dataclass
class _SpeechWindow:
    """User speech edges on the monotonic clock. The speech window resets at each commit."""

    speaking_since: float | None = None
    silence_since: float | None = None
    uncommitted: bool = False
    """Speech started after the last commit, so no turn has claimed it yet."""
    duration: float = 0.0
    """Speech time accumulated for the next commit."""

    @property
    def speaking(self) -> bool:
        return self.speaking_since is not None

    def started(self, at: float) -> None:
        self.speaking_since = at
        self.silence_since = None
        self.uncommitted = True

    def ended(self, at: float) -> None:
        if self.speaking_since is not None:
            self.duration += max(0.0, at - self.speaking_since)
        self.speaking_since = None
        self.silence_since = at

    def commit(self, now: float, eot_delay: float) -> float:
        """Close the turn's speech window and return its speech duration."""
        if self.speaking_since is None and not self.uncommitted:
            # No speech edge since the last commit: the EOT delay is the freshest anchor.
            self.silence_since = now - max(0.0, eot_delay)
        self.uncommitted = False
        duration, self.duration = self.duration, 0.0
        if self.speaking_since is not None:
            duration += now - self.speaking_since
            self.speaking_since = now
        return duration

    def release_at(self, threshold: float) -> float | None:
        """When ``threshold`` seconds of silence will have passed, or None while speech is open."""
        if self.speaking or self.silence_since is None:
            return None
        return self.silence_since + threshold


class _Turns:
    """Committed turns, the one inference in flight, and the one held prediction.

    Knows nothing about stages or the run lifecycle. AMDFSM decides what a settled
    prediction means; this class tracks which turn it belongs to and when it is due.
    """

    def __init__(self, inference_timeout: float) -> None:
        self._inference_timeout = inference_timeout
        self._turns: dict[int, AMDTurnHandle] = {}
        self._inference: AMDInference | None = None
        self._hold: AMDHold | None = None
        self._pending_dtmf_digits = ""

    def __contains__(self, turn_id: object) -> bool:
        return turn_id in self._turns

    @property
    def turn_id(self) -> int:
        return next(reversed(self._turns), 0)

    @property
    def inference(self) -> AMDInference | None:
        return self._inference

    @property
    def hold(self) -> AMDHold | None:
        return self._hold

    @property
    def settled(self) -> bool:
        """Whether no turn waits for a prediction or a held release."""
        if self._hold is not None:
            return False
        return not self._turns or self.prediction(self.turn_id) is not None

    def prediction(self, turn_id: int) -> AMDPredictionEvent | None:
        turn = self._turns[turn_id]
        while turn.prediction is None and turn.reuses is not None:
            turn = self._turns[turn.reuses]
            # A saved fallback does not release turns waiting for a held model result.
            if self._hold is not None and self._hold.turn is turn:
                return None
        return turn.prediction

    def record_prediction(self, turn: AMDTurnHandle, event: AMDPredictionEvent) -> None:
        """Save the first released prediction for the turn and any turns reusing it."""
        if turn.prediction is None:
            turn.prediction = event
        for reused in self._turns.values():
            if reused.prediction is None and reused.reuses is not None:
                reused.prediction = self.prediction(reused.turn_id)

    def dtmf_sent(self, digits: str) -> None:
        if not digits or any(digit not in "0123456789*#ABCD" for digit in digits):
            raise ValueError("digits must contain only 0-9, *, #, or A-D")
        self._pending_dtmf_digits += digits

    def commit(
        self, transcript: AMDTranscript, speech_duration: float, now: float
    ) -> AMDTurnHandle:
        turn = AMDTurnHandle(
            turn_id=self.turn_id + 1,
            committed_at=now,
            transcript=transcript,
            speech_duration=speech_duration,
            dtmf_digits=self._pending_dtmf_digits,
        )
        self._turns[turn.turn_id] = turn
        self._pending_dtmf_digits = ""
        return turn

    def outstanding(self) -> AMDTurnHandle | None:
        """The turn whose hold or live inference an empty turn can reuse."""
        if self._hold is not None:
            return self._hold.turn
        if self._inference is not None and not self._inference.timed_out:
            return self._inference.turn
        return None

    def classify(
        self, turn: AMDTurnHandle, now: float, *, stage: AMDCategory, allowed: list[AMDCategory]
    ) -> AMDClassifyRequest:
        """Start the turn's inference. A new request supersedes any hold."""
        self._hold = None
        self._inference = AMDInference(turn, deadline=now + self._inference_timeout)
        earlier_turns = [t for t in self._turns.values() if t.turn_id < turn.turn_id]
        return AMDClassifyRequest(
            stage=stage,
            allowed_next_categories=allowed,
            current_turn=turn.context(),
            earlier_turns=[t.context() for t in earlier_turns[-_HISTORY_LIMIT:]],
            speech_duration=turn.speech_duration,
        )

    def take_inference(self, turn_id: int) -> AMDInference | None:
        """Close the in-flight inference if it belongs to the turn."""
        inference = self._inference
        if inference is None or inference.turn.turn_id != turn_id:
            return None
        self._inference = None
        return inference

    def time_out_inference(self) -> AMDInference | None:
        """Mark the in-flight inference as timed out. It stays open for a late result."""
        inference = self._inference
        if inference is None or inference.timed_out:
            return None
        inference.timed_out = True
        return inference

    def hold_prediction(
        self, turn: AMDTurnHandle, prediction: AMDPrediction, release_at: float | None
    ) -> None:
        self._hold = AMDHold(turn, prediction, release_at)

    def freeze_hold(self) -> None:
        """Stop the hold's release timer while the user speaks."""
        if self._hold is not None:
            self._hold.release_at = None

    def take_hold(self) -> AMDHold | None:
        hold, self._hold = self._hold, None
        return hold

    def reset(self) -> None:
        self._inference = None
        self._hold = None
        self._pending_dtmf_digits = ""


class AMDFSM:
    """AMD state and policy, with explicit timestamps and no async resources.

    At most one inference and one held prediction exist at a time. A new turn with
    a transcript supersedes both. An empty turn reuses them instead.
    Normal path: commit -> inference -> [hold] -> publish -> transition.
    Machine predictions hold until the user has been silent long enough.
    New speech pauses a hold; speech end restarts its silence wait without requiring a commit.
    Timeouts and failures publish fallback predictions that keep the current stage.
    A late model prediction can update the stage without replacing the turn's saved prediction.
    Only the latest turn can reply; AMDReplyDecision selects its stage instructions.
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
        max_inference_timeouts: int,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._voicemail_idle_timeout = voicemail_idle_timeout
        self._timeout = timeout
        self._machine_silence_threshold = machine_silence_threshold
        self._max_uncertain_turns = max_uncertain_turns
        self._max_inference_timeouts = max_inference_timeouts

        self.lifecycle = AMDLifecycle.INITIALIZED
        self._user_speech = _SpeechWindow()
        self._turns = _Turns(inference_timeout)

        self._completion_reason = AMDReason.CANCELLED
        self._category = AMDCategory.UNCERTAIN
        self._previous_turn: AMDCategory | None = None
        self._previous_stage: AMDCategory | None = None
        self._latest: AMDPredictionEvent | None = None
        self._pending_voicemail_turn_id: int | None = None
        self._voicemail_reply_committed = False
        self._voicemail_message_played = False
        self._uncertain_turns = 0
        self._inference_timeouts = 0
        self._hard_deadline: float | None = None
        self._idle_deadline: float | None = None

    @property
    def category(self) -> AMDCategory:
        return self._category

    @property
    def turn_id(self) -> int:
        return self._turns.turn_id

    @property
    def voicemail_message_played(self) -> bool:
        return self._voicemail_message_played

    @property
    def next_deadline(self) -> float | None:
        return min((at for at, _ in self._deadlines()), default=None)

    def _deadlines(self) -> list[_Deadline]:
        """Armed deadlines with their actions. List order breaks ties: hard, idle, turn work."""
        deadlines: list[_Deadline] = []
        if self._hard_deadline is not None:
            deadlines.append((self._hard_deadline, lambda _: [self.finish(AMDReason.TIMEOUT)]))
        if self._idle_deadline is not None:
            deadlines.append((self._idle_deadline, lambda _: [self.finish(AMDReason.IDLE_TIMEOUT)]))
        hold = self._turns.hold
        if hold is not None and hold.release_at is not None:
            deadlines.append((hold.release_at, self._release_hold))
        inference = self._turns.inference
        if inference is not None and not inference.timed_out:
            deadlines.append((inference.deadline, self._time_out_inference))
        return deadlines

    def has_turn(self, turn_id: int) -> bool:
        return turn_id in self._turns

    def prediction(self, turn_id: int) -> AMDPredictionEvent | None:
        return self._turns.prediction(turn_id)

    def enter(self) -> None:
        if self.lifecycle is not AMDLifecycle.INITIALIZED:
            raise RuntimeError("use a new AMD instance for each run")
        self.lifecycle = AMDLifecycle.PENDING

    def start(self, now: float) -> None:
        if self.lifecycle is AMDLifecycle.PENDING:
            self.lifecycle = AMDLifecycle.ACTIVE
            self._hard_deadline = now + self._timeout

    def dtmf_sent(self, digits: str) -> None:
        if self.lifecycle in {AMDLifecycle.PENDING, AMDLifecycle.ACTIVE}:
            self._turns.dtmf_sent(digits)

    def speech_started(self, now: float) -> list[AMDEffect]:
        if self.lifecycle is not AMDLifecycle.ACTIVE:
            return []
        self._user_speech.started(now)
        self._idle_deadline = None
        self._turns.freeze_hold()
        return []

    def speech_ended(self, now: float, silence_duration: float) -> list[AMDEffect]:
        if self.lifecycle is not AMDLifecycle.ACTIVE:
            return []
        self._user_speech.ended(now - max(0.0, silence_duration))
        return self._resume_hold(now)

    def commit_turn(
        self, transcript: AMDTranscript, now: float, eot_delay: float
    ) -> list[AMDEffect]:
        if self.lifecycle is not AMDLifecycle.ACTIVE:
            raise RuntimeError("AMD must be listening before committing a turn")
        self._pending_voicemail_turn_id = None
        self._idle_deadline = None
        speech_duration = self._user_speech.commit(now, eot_delay)
        turn = self._turns.commit(transcript, speech_duration, now)
        if turn.transcript.transcript:
            request = self._turns.classify(
                turn, now, stage=self._category, allowed=sorted(ALLOWED[self._category])
            )
            return [request]

        # An empty turn reuses outstanding work, or settles with the current stage.
        outstanding = self._turns.outstanding()
        if outstanding is None:
            return self._settle(turn, AMDPrediction(self._category, AMDReason.REUSED), now)
        turn.reuses = outstanding.turn_id
        return self._resume_hold(now)

    def prediction_received(
        self, turn_id: int, category: AMDCategory, now: float, inference_duration: float
    ) -> list[AMDEffect]:
        inference = self._turns.take_inference(turn_id)
        if inference is None:
            return []
        category = self._category if category == AMDCategory.UNCERTAIN else category
        if category not in ALLOWED[self._category]:
            if inference.timed_out:
                return []
            prediction = AMDPrediction(self._category, AMDReason.INFERENCE_ERROR)
            return self._settle(inference.turn, prediction, now)
        self._inference_timeouts = 0
        reason = AMDReason.LATE_PREDICTION if inference.timed_out else AMDReason.PREDICTION
        prediction = AMDPrediction(category, reason, inference_duration=inference_duration)
        return self._settle(inference.turn, prediction, now)

    def inference_failed(self, turn_id: int, now: float) -> list[AMDEffect]:
        inference = self._turns.take_inference(turn_id)
        if inference is None or inference.timed_out:
            return []
        prediction = AMDPrediction(self._category, AMDReason.INFERENCE_ERROR)
        return self._settle(inference.turn, prediction, now)

    def deadline_reached(self, now: float) -> list[AMDEffect]:
        """Apply every due deadline in time order, even when the timer wakes late."""
        effects: list[AMDEffect] = []
        while due := [deadline for deadline in self._deadlines() if deadline[0] <= now]:
            _, action = min(due, key=lambda deadline: deadline[0])
            effects.extend(action(now))
        return effects

    def _resume_hold(self, now: float) -> list[AMDEffect]:
        hold = self._turns.hold
        if hold is None:
            return []
        hold.release_at = self._user_speech.release_at(self._machine_silence_threshold)
        if hold.release_at is not None and hold.release_at <= now:
            return self._release_hold(now)
        return []

    def _release_hold(self, now: float) -> list[AMDEffect]:
        hold = self._turns.take_hold()
        if hold is None:
            return []
        return self._publish(hold.turn, hold.prediction, now)

    def _time_out_inference(self, now: float) -> list[AMDEffect]:
        inference = self._turns.time_out_inference()
        if inference is None:
            return []
        self._inference_timeouts += 1
        prediction = AMDPrediction(self._category, AMDReason.INFERENCE_TIMEOUT)
        return self._settle(inference.turn, prediction, now)

    def update_idle(self, now: float, *, session_busy: bool) -> None:
        idle = (
            self.lifecycle is AMDLifecycle.ACTIVE
            and not session_busy
            and not self._user_speech.speaking
            and self._turns.settled
        )
        if not idle:
            self._idle_deadline = None
        elif self._idle_deadline is None:
            timeout = (
                self._voicemail_idle_timeout
                if self._category == AMDCategory.MACHINE_VM
                else self._idle_timeout
            )
            self._idle_deadline = now + timeout

    def authorize_reply(self, turn_id: int) -> AMDReplyDecision:
        """Reserve a reply for the turn and select its stage instructions."""
        finished = self.lifecycle is AMDLifecycle.FINISHED
        if finished and self._category is AMDCategory.MACHINE_UNAVAILABLE:
            return _SKIP
        if turn_id not in self._turns:
            return _ABSTAIN
        if turn_id != self.turn_id:
            return _SKIP
        if finished:
            human_after_machine = (
                self._category is AMDCategory.HUMAN and self._previous_stage in MACHINE
            )
            return AMDReplyDecision(
                allow=True, instructions_for=AMDCategory.HUMAN if human_after_machine else None
            )
        if self.prediction(turn_id) is None:
            return _SKIP
        if self._category is AMDCategory.UNCERTAIN:
            return _ABSTAIN
        if self._category is AMDCategory.MACHINE_VM:
            if self._voicemail_reply_committed or self._pending_voicemail_turn_id is not None:
                return _SKIP
            self._pending_voicemail_turn_id = turn_id
            return AMDReplyDecision(
                allow=True, instructions_for=self._category, track_voicemail=True
            )
        return AMDReplyDecision(allow=True, instructions_for=self._category)

    def commit_voicemail_reply(self, turn_id: int) -> bool:
        """Consume the pending reservation when its reply handle is accepted."""
        if self.lifecycle is not AMDLifecycle.ACTIVE or self._pending_voicemail_turn_id != turn_id:
            return False
        self._pending_voicemail_turn_id = None
        self._voicemail_reply_committed = True
        return True

    def voicemail_played(self) -> None:
        self._voicemail_message_played = True

    def finish(self, reason: AMDReason) -> AMDCompletedEvent:
        if self.lifecycle is AMDLifecycle.FINISHED:
            return self.completion()
        self.lifecycle = AMDLifecycle.FINISHED
        self._completion_reason = reason
        self._hard_deadline = self._idle_deadline = None
        self._turns.reset()
        return self.completion()

    def completion(self) -> AMDCompletedEvent:
        if self.lifecycle is not AMDLifecycle.FINISHED:
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

    def _settle(
        self, turn: AMDTurnHandle, prediction: AMDPrediction, now: float
    ) -> list[AMDEffect]:
        """Hold a machine prediction until enough user silence, otherwise publish it."""
        if self.lifecycle is AMDLifecycle.FINISHED:
            return []
        if not prediction.from_model and turn.prediction is not None:
            return []  # the turn already settled; a late fallback adds nothing
        hold = self._turns.hold
        if hold is not None and hold.turn is turn:
            self._turns.take_hold()
        if prediction.category in MACHINE and self._machine_silence_threshold > 0:
            self._idle_deadline = None
            if hold is not None and hold.turn is not turn and prediction.from_model:
                hold.turn.reuses = turn.turn_id
                hold.turn = turn
                hold.prediction = prediction
                if hold.release_at is not None and hold.release_at <= now:
                    return self._release_hold(now)
                return []
            release_at = self._user_speech.release_at(self._machine_silence_threshold)
            if release_at is None or now < release_at:
                if self._turns.hold is None:
                    self._turns.hold_prediction(turn, prediction, release_at)
                return []
        return self._publish(turn, prediction, now)

    def _publish(
        self, turn: AMDTurnHandle, prediction: AMDPrediction, now: float
    ) -> list[AMDEffect]:
        """Record the prediction on its turn, emit it, and apply its effect on the run."""
        previous = self._latest
        if prediction.from_model and prediction.category != self._category:
            self._previous_stage = self._category
            self._category = prediction.category
        event = self._make_prediction_event(turn, prediction, now)
        self._turns.record_prediction(turn, event)
        if prediction.reason is AMDReason.REUSED:
            return []
        self._latest = event
        if prediction.from_model:
            completed = self._transition(previous_turn=previous.category if previous else None)
        elif (
            prediction.reason is AMDReason.INFERENCE_TIMEOUT
            and self._inference_timeouts >= self._max_inference_timeouts
        ):
            completed = self.finish(AMDReason.INFERENCE_TIMEOUT)
        else:
            completed = None
        effects: list[AMDEffect] = [event.model_copy()]
        if completed is not None:
            effects.append(completed)
        elif prediction.from_model and prediction.category is AMDCategory.MACHINE_IVR:
            # Prediction listeners can commit a newer turn and cancel this menu work.
            effects.insert(0, AMDMenuRequest(turn.turn_id, turn.transcript.transcript))
        return effects

    def _transition(self, *, previous_turn: AMDCategory | None) -> AMDCompletedEvent | None:
        if self._category != (previous_turn or AMDCategory.UNCERTAIN):
            self._idle_deadline = None
            self._pending_voicemail_turn_id = None
            self._voicemail_reply_committed = False
        self._previous_turn = previous_turn
        if self._category in TERMINAL:
            return self.finish(AMDReason.FINISHED)
        self._uncertain_turns = (
            self._uncertain_turns + 1 if self._category is AMDCategory.UNCERTAIN else 0
        )
        if self._uncertain_turns >= self._max_uncertain_turns:
            return self.finish(AMDReason.MAX_UNCERTAIN_TURNS)
        return None

    def _make_prediction_event(
        self, turn: AMDTurnHandle, prediction: AMDPrediction, now: float
    ) -> AMDPredictionEvent:
        return AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=prediction.category,
            reason=prediction.reason,
            transcript=turn.transcript.transcript,
            speech_duration=turn.speech_duration,
            delay=now - turn.committed_at,
            inference_duration=prediction.inference_duration,
            prev_turn_category=self._latest.category if self._latest else None,
            prev_stage_category=self._previous_stage,
            voicemail_message_played=self._voicemail_message_played,
        )
