import time

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given


class ExponentialMovingAverage:
    def __init__(self, alpha: float, initial: float | None = None) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self._alpha = alpha
        self._value: float | None = initial

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, sample: float) -> float:
        if self._value is None:
            self._value = sample
        else:
            self._value = self._alpha * sample + (1 - self._alpha) * self._value
        return self._value

    def reset(self, value: float | None = None) -> None:
        self._value = value


class DynamicEndpointing:
    def __init__(self, min_delay: float, max_delay: float, alpha: float = 0.1):
        """
        Dynamically adjust the endpointing delay based on the speech activity.

        Args:
            min_delay: Minimum delay in seconds.
            max_delay: Maximum delay in seconds.
            alpha: Exponential moving average coefficient.

        The endpointing delay is adjusted based on the following information:

        1. Pauses between utterances:

        [utterance] [pause] [utterance] [pause] [utterance] (<- min delay should cover this)

        2. Pauses between an utterance and next immediate interruption:

        [utterance] [   pause   ] [immediate interruption] (<- this should be a false EOT, and min delay should cover this)
                        [agent speech interrupted]

        3. Pauses between a user utterance and agent speech:

        [utterance] [pause]                  (<- max delay should cover this)
                           [agent speech]    (this could be interrupted later, but that would be the next turn)
        """

        self._min_delay = min_delay
        self._max_delay = max_delay

        self._utterance_pause = ExponentialMovingAverage(alpha=alpha, initial=min_delay)
        self._turn_pause = ExponentialMovingAverage(alpha=alpha, initial=max_delay)

        self._utterance_started_at: float | None = None
        self._agent_speech_started_at: float | None = None
        self._utterance_ended_at: float | None = None
        self._interrupting = False

    @property
    def min_delay(self) -> float:
        return (
            self._utterance_pause.value
            if self._utterance_pause.value is not None
            else self._min_delay
        )

    @property
    def max_delay(self) -> float:
        return self._turn_pause.value if self._turn_pause.value is not None else self._max_delay

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

    def on_agent_speech_started(self, adjustment: float = 0.0) -> None:
        self._agent_speech_started_at = time.time() + adjustment
        logger.debug(
            f"agent speech started at: {self._agent_speech_started_at}",
            extra={
                "adjustment": adjustment,
            },
        )

    def on_utterance_started(self, adjustment: float = 0.0, interruption: bool = False) -> None:
        self._utterance_started_at = time.time() + adjustment
        logger.debug(
            f"utterance started at: {self._utterance_started_at}",
            extra={
                "adjustment": adjustment,
                "interruption": interruption,
                "interrupting": self._interrupting,
            },
        )

        if interruption and not self._interrupting:
            # If this is an immediate interruption, update the min delay (case 2)
            turn_delay, interruption_delay = self.immediate_interruption_delay
            if (
                (0 < interruption_delay <= self.min_delay)
                and (0 < turn_delay <= self.max_delay)
                and (pause := self.between_utterance_delay) > 0
            ):
                prev_val = self.min_delay
                self._utterance_pause.update(min(max(pause, self._min_delay), self._max_delay))
                logger.debug(
                    f"min endpointing delay updated: {prev_val} -> {self.min_delay}",
                    extra={
                        "reason": "immediate interruption",
                        "interruption": interruption,
                        "pause": pause,
                        "interruption_delay": interruption_delay,
                        "turn_delay": turn_delay,
                        "max_delay": self.max_delay,
                        "min_delay": self.min_delay,
                    },
                )
            # If this is not an immediate interruption, update the max delay (case 3)
            elif (pause := self.between_turn_delay) > 0:
                prev_val = self.max_delay
                self._turn_pause.update(min(max(pause, self.min_delay), self._max_delay))
                logger.debug(
                    f"max endpointing delay updated: {prev_val} -> {self.max_delay}",
                    extra={
                        "reason": "EOT (interruption)",
                        "interruption": interruption,
                        "pause": pause,
                        "max_delay": self.max_delay,
                        "min_delay": self.min_delay,
                        "between_utterance_delay": self.between_utterance_delay,
                        "between_turn_delay": self.between_turn_delay,
                    },
                )

            self._agent_speech_started_at = None
            self._interrupting = True
            return

        if interruption and self._interrupting:
            # duplicate calls from _interrupt_by_audio_activity and on_start_of_speech
            return

        if (pause := self.between_utterance_delay) > 0 and self._agent_speech_started_at is None:
            prev_val = self.min_delay
            self._utterance_pause.update(min(max(pause, self._min_delay), self._max_delay))
            logger.debug(
                f"min endpointing delay updated: {prev_val} -> {self.min_delay}",
                extra={
                    "reason": "pause between utterances (case 1)",
                    "interruption": interruption,
                    "pause": pause,
                    "max_delay": self.max_delay,
                    "min_delay": self.min_delay,
                    "interrupting": self._interrupting,
                },
            )
        elif (pause := self.between_turn_delay) > 0:
            prev_val = self.max_delay
            self._turn_pause.update(min(max(pause, self.min_delay), self._max_delay))
            logger.debug(
                f"max endpointing delay updated due to pause: {prev_val} -> {self.max_delay}",
                extra={
                    "reason": "new turn (case 3)",
                    "pause": pause,
                    "max_delay": self.max_delay,
                    "min_delay": self.min_delay,
                },
            )

        self._agent_speech_started_at = None
        self._interrupting = False

    def on_utterance_ended(self, adjustment: float = 0.0) -> None:
        self._utterance_ended_at = time.time() + adjustment
        logger.debug(
            f"utterance ended at: {self._utterance_ended_at}",
            extra={
                "adjustment": adjustment,
                "interrupting": self._interrupting,
                "max_delay": self.max_delay,
                "min_delay": self.min_delay,
            },
        )
        # Reset the interrupting flag since we only need to track the first interruption utterance
        self._interrupting = False

    def update_options(
        self, *, min_delay: NotGivenOr[float] = NOT_GIVEN, max_delay: NotGivenOr[float] = NOT_GIVEN
    ) -> None:
        if is_given(min_delay):
            self._min_delay = min_delay
            self._utterance_pause.reset(self._min_delay)

        if is_given(max_delay):
            self._max_delay = max_delay
            self._turn_pause.reset(self._max_delay)
