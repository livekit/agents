"""Sans-IO endpointing.

Invariants: imports only the stdlib; time enters only through event ``at`` fields (seconds);
never logs or performs I/O, every observable fact is returned from ``handle``.
"""

from __future__ import annotations

from dataclasses import dataclass

AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD = 0.25  # seconds


@dataclass(frozen=True)
class UserSpeechStarted:
    at: float
    overlapping: bool = False


@dataclass(frozen=True)
class UserSpeechEnded:
    at: float
    interruption: bool | None = None


@dataclass(frozen=True)
class AgentSpeechStarted:
    at: float


@dataclass(frozen=True)
class AgentSpeechEnded:
    at: float


@dataclass(frozen=True)
class OptionsUpdated:
    """``None`` leaves the option unchanged."""

    min_delay: float | None = None
    max_delay: float | None = None
    alpha: float | None = None


EndpointingEvent = (
    UserSpeechStarted | UserSpeechEnded | AgentSpeechStarted | AgentSpeechEnded | OptionsUpdated
)


@dataclass(frozen=True)
class MinDelayUpdated:
    """``turn_delay`` and ``interruption_delay`` are set only for ``"immediate interruption"``."""

    reason: str
    prev: float
    new: float
    pause: float
    max_delay: float
    turn_delay: float | None
    interruption_delay: float | None


@dataclass(frozen=True)
class UtteranceEndAdjusted:
    at: float


@dataclass(frozen=True)
class NonInterruptionOverridden:
    gap: float


EndpointingOutput = MinDelayUpdated | UtteranceEndAdjusted | NonInterruptionOverridden


class FixedEndpointing:
    def __init__(self, min_delay: float, max_delay: float) -> None:
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._overlapping = False

    @property
    def min_delay(self) -> float:
        return self._min_delay

    @property
    def max_delay(self) -> float:
        return self._max_delay

    @property
    def overlapping(self) -> bool:
        return self._overlapping

    def handle(self, event: EndpointingEvent) -> list[EndpointingOutput]:
        if isinstance(event, UserSpeechStarted):
            self._overlapping = event.overlapping
        elif isinstance(event, UserSpeechEnded):
            self._overlapping = False
        elif isinstance(event, OptionsUpdated):
            if event.min_delay is not None:
                self._min_delay = event.min_delay
            if event.max_delay is not None:
                self._max_delay = event.max_delay
        return []


class _ExpFilter:
    """Clamped exponential moving average; the value always stays in [min_val, max_val]."""

    def __init__(self, alpha: float, initial: float, min_val: float, max_val: float) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self._alpha = alpha
        self._filtered = initial
        self._min_val = min_val
        self._max_val = max_val

    @property
    def value(self) -> float:
        return self._filtered

    def reset(
        self,
        *,
        alpha: float | None = None,
        initial: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        if alpha is not None:
            assert 0 < alpha <= 1, "alpha must be in (0, 1]."
            self._alpha = alpha
        if initial is not None:
            self._filtered = initial
        if min_val is not None:
            self._min_val = min_val
        if max_val is not None:
            self._max_val = max_val

    def apply(self, sample: float) -> float:
        filtered = self._alpha * self._filtered + (1 - self._alpha) * sample
        self._filtered = max(min(filtered, self._max_val), self._min_val)
        return self._filtered


class DynamicEndpointing(FixedEndpointing):
    def __init__(self, min_delay: float, max_delay: float, alpha: float = 0.9) -> None:
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

        self._utterance_pause = _ExpFilter(
            alpha=alpha, initial=min_delay, min_val=min_delay, max_val=max_delay
        )

        self._utterance_started_at: float | None = None
        self._utterance_ended_at: float | None = None
        self._agent_speech_started_at: float | None = None
        self._agent_speech_ended_at: float | None = None
        self._agent_speaking = False
        self._speaking = False

    @property
    def min_delay(self) -> float:
        return min(self._utterance_pause.value, self.max_delay)

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

    def handle(self, event: EndpointingEvent) -> list[EndpointingOutput]:
        if isinstance(event, UserSpeechStarted):
            return self._on_user_speech_started(event)
        if isinstance(event, UserSpeechEnded):
            return self._on_user_speech_ended(event)
        if isinstance(event, AgentSpeechStarted):
            return self._on_agent_speech_started(event)
        if isinstance(event, AgentSpeechEnded):
            return self._on_agent_speech_ended(event)
        return self._on_options_updated(event)

    def _on_agent_speech_started(self, event: AgentSpeechStarted) -> list[EndpointingOutput]:
        outputs: list[EndpointingOutput] = []
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
            self._utterance_ended_at = event.at - 1e-3
            outputs.append(UtteranceEndAdjusted(at=self._utterance_ended_at))

        self._agent_speech_started_at = event.at
        self._agent_speech_ended_at = None
        self._agent_speaking = True
        self._overlapping = self._speaking
        return outputs

    def _on_agent_speech_ended(self, event: AgentSpeechEnded) -> list[EndpointingOutput]:
        # Keep the agent speech timestamps until the next user utterance ends so
        # the pause across an agent turn is not learned as an intra-user pause.
        if self._agent_speaking:
            self._agent_speech_ended_at = event.at
        self._agent_speaking = False
        self._overlapping = False
        return []

    def _on_user_speech_started(self, event: UserSpeechStarted) -> list[EndpointingOutput]:
        if self._overlapping:
            # duplicate calls from _interrupt_by_audio_activity and on_start_of_speech
            return []

        self._utterance_started_at = event.at
        self._overlapping = event.overlapping
        self._speaking = True
        return []

    def _on_user_speech_ended(self, event: UserSpeechEnded) -> list[EndpointingOutput]:
        outputs: list[EndpointingOutput] = []
        if event.interruption is False and self._overlapping:
            # If user speech started within AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD of agent speech,
            # don't skip — TTS leading silence can cause the agent speech timestamp
            # to precede actual audible audio, making this look like a backchannel
            # when it's really the user speaking before hearing the agent.
            if (
                self._utterance_started_at is not None
                and self._agent_speech_started_at is not None
                and abs(self._utterance_started_at - self._agent_speech_started_at)
                < AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD
            ):
                outputs.append(
                    NonInterruptionOverridden(
                        gap=abs(self._utterance_started_at - self._agent_speech_started_at)
                    )
                )
            else:
                # skip update for a confirmed non-interruption, such as a backchannel
                self._overlapping = False
                self._speaking = False
                self._utterance_started_at = None
                self._utterance_ended_at = None
                return outputs

        if self._overlapping or (
            self._agent_speech_started_at is not None and self._agent_speech_ended_at is None
        ):  # this is an interruption (agent is still speaking)
            # If this is an immediate interruption, update the min delay (case 2)
            turn_delay, interruption_delay = self.immediate_interruption_delay
            if (
                (0 < interruption_delay <= self.min_delay)
                and (0 < turn_delay <= self.max_delay)
                and (pause := self.between_utterance_delay) > 0
            ):
                prev_val = self.min_delay
                self._utterance_pause.apply(pause)
                outputs.append(
                    MinDelayUpdated(
                        reason="immediate interruption",
                        prev=prev_val,
                        new=self.min_delay,
                        pause=pause,
                        max_delay=self.max_delay,
                        turn_delay=turn_delay,
                        interruption_delay=interruption_delay,
                    )
                )

        else:
            # Only learn an intra-user pause when no agent turn occurred between utterances.
            if (
                (pause := self.between_utterance_delay) > 0
                and self._agent_speech_ended_at is None
                and self._agent_speech_started_at is None
            ):
                prev_val = self.min_delay
                self._utterance_pause.apply(pause)
                outputs.append(
                    MinDelayUpdated(
                        reason="pause between utterances",
                        prev=prev_val,
                        new=self.min_delay,
                        pause=pause,
                        max_delay=self.max_delay,
                        turn_delay=None,
                        interruption_delay=None,
                    )
                )

        self._utterance_ended_at = event.at
        # Preserve an active agent interval until its end is recorded.
        if not self._agent_speaking:
            self._agent_speech_started_at = None
            self._agent_speech_ended_at = None
        self._speaking = False
        self._overlapping = False
        return outputs

    def _on_options_updated(self, event: OptionsUpdated) -> list[EndpointingOutput]:
        if event.min_delay is not None:
            self._min_delay = event.min_delay
            self._utterance_pause.reset(initial=self._min_delay, min_val=self._min_delay)

        if event.max_delay is not None:
            self._max_delay = event.max_delay
            self._utterance_pause.reset(max_val=self._max_delay)

        if event.alpha is not None:
            self._utterance_pause.reset(alpha=event.alpha)
        return []
