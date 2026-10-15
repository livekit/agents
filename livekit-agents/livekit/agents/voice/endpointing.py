from dataclasses import dataclass, field
from typing import NamedTuple

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from ..utils.exp_filter import ExpFilter
from .turn import EndpointingOptions

_AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD = 0.25  # seconds


class _InterruptionPause(NamedTuple):
    pause: float
    turn_delay: float
    interruption_delay: float


@dataclass
class _PendingPause:
    interruption_pause: _InterruptionPause | None
    utterance_pauses: list[float] = field(default_factory=list)


class BaseEndpointing:
    def __init__(self, min_delay: float, max_delay: float):
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._overlapping = False

    def update_options(
        self, *, min_delay: NotGivenOr[float] = NOT_GIVEN, max_delay: NotGivenOr[float] = NOT_GIVEN
    ) -> None:
        if is_given(min_delay):
            self._min_delay = min_delay
        if is_given(max_delay):
            self._max_delay = max_delay

    @property
    def min_delay(self) -> float:
        return self._min_delay

    @property
    def max_delay(self) -> float:
        return self._max_delay

    @property
    def overlapping(self) -> bool:
        return self._overlapping

    def on_start_of_speech(self, started_at: float, overlapping: bool = False) -> None:
        self._overlapping = overlapping

    def on_end_of_speech(self, ended_at: float, interruption: NotGivenOr[bool] = NOT_GIVEN) -> None:
        self._overlapping = False

    def on_start_of_agent_speech(self, started_at: float, *, resumed: bool = False) -> None:
        pass

    def on_end_of_agent_speech(self, ended_at: float, *, paused: bool = False) -> None:
        pass

    def on_pause_resolved(self, *, interrupted: bool) -> None:
        """Resolve a provisional speech pause, independently of audible playout."""
        pass


class DynamicEndpointing(BaseEndpointing):
    def __init__(self, min_delay: float, max_delay: float, alpha: float = 0.9):
        """
        Dynamically adjust the endpointing delay based on the speech activity.

        Args:
            min_delay: Minimum delay in seconds (the learned floor).
            max_delay: Maximum delay in seconds (a fixed ceiling).
            alpha: Exponential moving average coefficient. The higher the value, the more weight is given to the history. Defaults to 0.9.

        ``min_delay`` is learned from the user's pausing behavior so we don't cut
        off a user who pauses mid-turn. It adapts from the following pauses:

        1. Pauses between utterances:

        [utterance] [pause] [utterance] [pause] [utterance] (<- min delay should cover this)

        2. Pauses between an utterance and next immediate interruption:

        [utterance] [   pause   ] [immediate interruption] (<- this should be a false EOT, and min delay should cover this)
                        [agent speech interrupted]

        ``max_delay`` is a fixed ceiling, not learned. There is no natural "lower" max_delay to observe
        in a session. It only clamps the learned ``min_delay`` from above."""

        super().__init__(min_delay=min_delay, max_delay=max_delay)

        self._utterance_pause = ExpFilter(
            alpha=alpha, initial=min_delay, min_val=min_delay, max_val=max_delay
        )

        self._utterance_started_at: float | None = None
        self._utterance_ended_at: float | None = None
        self._agent_speech_started_at: float | None = None
        self._agent_speech_ended_at: float | None = None
        self._agent_speaking = False
        self._speaking = False
        self._pending_pause: _PendingPause | None = None

    @property
    def min_delay(self) -> float:
        return min(
            self._utterance_pause.value
            if self._utterance_pause.value is not None
            else self._min_delay,
            self.max_delay,
        )

    @property
    def between_utterance_delay(self) -> float:
        if self._utterance_ended_at is None:
            return 0.0
        if self._utterance_started_at is None:
            return 0.0

        return max(0, self._utterance_started_at - self._utterance_ended_at)

    @property
    def between_turn_delay(self) -> float:
        if self._agent_speech_started_at is None:
            return 0.0
        if self._utterance_ended_at is None:
            return 0.0

        return max(0, self._agent_speech_started_at - self._utterance_ended_at)

    @property
    def immediate_interruption_delay(self) -> tuple[float, float]:
        """
        Returns the two pauses in the following case:
        [utterance] [first val][second val] [immediate interruption]
                               [agent speech interrupted]
        """
        if self._utterance_started_at is None:
            return 0.0, 0.0
        if self._agent_speech_started_at is None:
            return 0.0, 0.0

        return (
            self.between_turn_delay,
            abs(self.between_utterance_delay - self.between_turn_delay),
        )

    def on_start_of_agent_speech(self, started_at: float, *, resumed: bool = False) -> None:
        if resumed:
            self.on_pause_resolved(interrupted=False)
        elif self._pending_pause is not None:
            logger.debug("discarding unresolved endpointing pause on new agent speech")
        self._pending_pause = None

        # Agent speech started before the current user utterance ended, so the stored
        # end still belongs to the previous utterance. Move it just before agent speech
        # to exclude this overlap from dynamic endpointing statistics.
        if (
            not self._agent_speaking
            and self._speaking
            and self._utterance_started_at is not None
            and self._utterance_ended_at is not None
            and self._utterance_ended_at < self._utterance_started_at
        ):
            self._utterance_ended_at = started_at - 1e-3
            logger.trace(
                "utterance ended at adjusted: %s",
                self._utterance_ended_at,
            )

        self._agent_speech_started_at = started_at
        self._agent_speech_ended_at = None
        self._agent_speaking = True
        self._overlapping = self._speaking

    def on_end_of_agent_speech(self, ended_at: float, *, paused: bool = False) -> None:
        # Keep the agent speech timestamps until the next user utterance ends so
        # the pause across an agent turn is not learned as an intra-user pause.
        if self._agent_speaking:
            if paused:
                candidate = self._get_interruption_pause() if self._speaking else None
                self._pending_pause = _PendingPause(interruption_pause=candidate)
            self._agent_speech_ended_at = ended_at
        self._agent_speaking = False
        self._overlapping = False

    def on_pause_resolved(self, *, interrupted: bool) -> None:
        pending, self._pending_pause = self._pending_pause, None
        if pending is None:
            return

        if interrupted:
            if (candidate := pending.interruption_pause) is not None:
                self._learn_pause(
                    candidate.pause,
                    reason="immediate interruption",
                    interruption_delay=candidate.interruption_delay,
                    turn_delay=candidate.turn_delay,
                )
            for pause in pending.utterance_pauses:
                self._learn_pause(pause, reason="pause between utterances")
        else:
            # not an interruption so we discard utterance history
            # including any ongoing utterance.
            self._utterance_started_at = None
            self._utterance_ended_at = None

        if not self._speaking:
            self._agent_speech_started_at = None
            self._agent_speech_ended_at = None

    def on_start_of_speech(self, started_at: float, overlapping: bool = False) -> None:
        if self._overlapping:
            # duplicate calls from _interrupt_by_audio_activity and on_start_of_speech
            return

        # a pause between two utterances during agent speech pause
        if (
            self._pending_pause is not None
            and not self._speaking
            and self._utterance_ended_at is not None
            and (pause := started_at - self._utterance_ended_at) > 0
        ):
            self._pending_pause.utterance_pauses.append(pause)

        self._utterance_started_at = started_at
        self._overlapping = overlapping
        self._speaking = True

    def on_end_of_speech(self, ended_at: float, interruption: NotGivenOr[bool] = NOT_GIVEN) -> None:
        # user finishes speaking when agent is already paused
        # record EOS now; defer learning until the speech pause is resolved.
        if self._pending_pause is not None:
            self._utterance_ended_at = ended_at
            self._speaking = False
            self._overlapping = False
            return

        if is_given(interruption) and not interruption and self._overlapping:
            # If user speech started within _AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD of agent speech,
            # don't skip — TTS leading silence can cause the agent speech timestamp
            # to precede actual audible audio, making this look like a backchannel
            # when it's really the user speaking before hearing the agent.
            if (
                self._utterance_started_at is not None
                and self._agent_speech_started_at is not None
                and abs(self._utterance_started_at - self._agent_speech_started_at)
                < _AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD
            ):
                logger.trace(
                    "overriding non-interruption verdict: user speech started within %.3fs of "
                    "agent speech "
                    "(within grace period of %.3fs)",
                    abs(self._utterance_started_at - self._agent_speech_started_at),
                    _AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD,
                )
            else:
                # skip update for a confirmed non-interruption, such as a backchannel
                self._overlapping = False
                self._speaking = False
                self._utterance_started_at = None
                self._utterance_ended_at = None
                return

        if self._overlapping or (
            self._agent_speech_started_at is not None and self._agent_speech_ended_at is None
        ):  # this is an interruption (agent is still speaking)
            # If this is an immediate interruption, update the min delay (case 2)
            if (candidate := self._get_interruption_pause()) is not None:
                self._learn_pause(
                    candidate.pause,
                    reason="immediate interruption",
                    interruption_delay=candidate.interruption_delay,
                    turn_delay=candidate.turn_delay,
                )

        else:
            # Only learn an intra-user pause when no agent turn occurred between utterances.
            if (
                (pause := self.between_utterance_delay) > 0
                and self._agent_speech_ended_at is None
                and self._agent_speech_started_at is None
            ):
                self._learn_pause(pause, reason="pause between utterances")

        self._utterance_ended_at = ended_at
        # Preserve an active agent interval until its end is recorded.
        if not self._agent_speaking:
            self._agent_speech_started_at = None
            self._agent_speech_ended_at = None
        self._speaking = False
        self._overlapping = False

    def _learn_pause(self, pause: float, *, reason: str, **timing: float) -> None:
        prev_val = self.min_delay
        self._utterance_pause.apply(1.0, pause)
        logger.debug(
            "min endpointing delay updated: %s -> %s",
            prev_val,
            self.min_delay,
            extra={
                "reason": reason,
                "pause": pause,
                **timing,
                "max_delay": self.max_delay,
                "min_delay": self.min_delay,
            },
        )

    def _get_interruption_pause(self) -> _InterruptionPause | None:
        turn_delay, interruption_delay = self.immediate_interruption_delay
        if (
            (0 < interruption_delay <= self.min_delay)
            and (0 < turn_delay <= self.max_delay)
            and (pause := self.between_utterance_delay) > 0
        ):
            return _InterruptionPause(pause, turn_delay, interruption_delay)
        return None

    def update_options(
        self,
        *,
        min_delay: NotGivenOr[float] = NOT_GIVEN,
        max_delay: NotGivenOr[float] = NOT_GIVEN,
        alpha: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(min_delay):
            self._min_delay = min_delay
            self._utterance_pause.reset(initial=self._min_delay, min_val=self._min_delay)

        if is_given(max_delay):
            self._max_delay = max_delay
            self._utterance_pause.reset(max_val=self._max_delay)

        if is_given(alpha):
            self._utterance_pause.reset(alpha=alpha)


def create_endpointing(options: EndpointingOptions) -> BaseEndpointing:
    match options.get("mode", "fixed"):
        case "dynamic":
            return DynamicEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
                alpha=options["alpha"],
            )
        case _:
            return BaseEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
            )
