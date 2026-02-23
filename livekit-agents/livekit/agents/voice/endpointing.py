import time

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from ..utils.exp_filter import ExpFilter


class DynamicEndpointing:
    def __init__(self, min_delay: float, max_delay: float, alpha: float = 0.9):
        """
        Dynamically adjust the endpointing delay based on the speech activity.

        Args:
            min_delay: Minimum delay in seconds.
            max_delay: Maximum delay in seconds.
            alpha: Exponential moving average coefficient. The higher the value, the more weight is given to the history. Defaults to 0.9.

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

        self._utterance_pause = ExpFilter(
            alpha=alpha, initial=min_delay, min_val=min_delay, max_val=max_delay
        )
        self._turn_pause = ExpFilter(
            alpha=alpha, initial=max_delay, min_val=min_delay, max_val=max_delay
        )

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
        if self._interrupting:
            # duplicate calls from _interrupt_by_audio_activity and on_start_of_speech
            return

        if (
            self._utterance_started_at is not None
            and self._utterance_ended_at is not None
            and self._utterance_ended_at < self._utterance_started_at
            and interruption
            and self._agent_speech_started_at is not None
        ):
            # VAD interrupt by audio activity is triggered before end of speech is detected
            # adjust the utterance ended time to be just before the agent speech started
            self._utterance_ended_at = self._agent_speech_started_at - 1e-3
            logger.debug(
                f"utterance ended at adjusted: {self._utterance_ended_at}",
            )

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
                self._utterance_pause.apply(1.0, pause)
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
                self._turn_pause.apply(1.0, pause)
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

        if (pause := self.between_utterance_delay) > 0 and self._agent_speech_started_at is None:
            prev_val = self.min_delay
            self._utterance_pause.apply(1.0, pause)
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
            self._turn_pause.apply(1.0, pause)
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
            self._utterance_pause.reset(initial=self._min_delay, min_val=self._min_delay)

        if is_given(max_delay):
            self._max_delay = max_delay
            self._turn_pause.reset(initial=self._max_delay, max_val=self._max_delay)
