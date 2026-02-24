from typing import TYPE_CHECKING

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from ..utils.exp_filter import ExpFilter

if TYPE_CHECKING:
    from .turn import EndpointingOptions


_AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD = 0.25  # seconds


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

    def on_end_of_speech(self, ended_at: float, should_ignore: bool = False) -> None:
        self._overlapping = False

    def on_start_of_agent_speech(self, started_at: float) -> None:
        pass

    def on_end_of_agent_speech(self, ended_at: float) -> None:
        pass


class DynamicEndpointing(BaseEndpointing):
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

        super().__init__(min_delay=min_delay, max_delay=max_delay)

        self._utterance_pause = ExpFilter(
            alpha=alpha, initial=min_delay, min_val=min_delay, max_val=max_delay
        )
        self._turn_pause = ExpFilter(
            alpha=alpha, initial=max_delay, min_val=min_delay, max_val=max_delay
        )

        self._utterance_started_at: float | None = None
        self._utterance_ended_at: float | None = None
        self._agent_speech_started_at: float | None = None
        self._agent_speech_ended_at: float | None = None
        self._speaking = False

    @property
    def min_delay(self) -> float:
        return (
            self._utterance_pause.value
            if self._utterance_pause.value is not None
            else self._min_delay
        )

    @property
    def max_delay(self) -> float:
        turn_val = self._turn_pause.value if self._turn_pause.value is not None else self._max_delay
        return max(turn_val, self.min_delay)

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

    def on_start_of_agent_speech(self, started_at: float) -> None:
        self._agent_speech_started_at = started_at
        self._agent_speech_ended_at = None
        self._overlapping = False
        # TODO: @chenghao-mou remove this debug log
        logger.debug(
            "agent speech started at: %s",
            self._agent_speech_started_at,
        )

    def on_end_of_agent_speech(self, ended_at: float) -> None:
        # NOTE: intentionally keep _agent_speech_started_at so that
        # between_turn_delay can be computed in the normal end-of-speech path
        self._agent_speech_ended_at = ended_at
        self._overlapping = False
        # TODO: @chenghao-mou remove this debug log
        logger.debug(
            "agent speech ended at: %s",
            ended_at,
        )

    def on_start_of_speech(self, started_at: float, overlapping: bool = False) -> None:
        if self._overlapping:
            # duplicate calls from _interrupt_by_audio_activity and on_start_of_speech
            return

        # VAD interrupt by audio activity is triggered before end of speech is detected
        # adjust the utterance ended time to be just before the agent speech started
        if (
            self._utterance_started_at is not None
            and self._utterance_ended_at is not None
            and self._agent_speech_started_at is not None
            and self._utterance_ended_at < self._utterance_started_at
            and overlapping
        ):
            self._utterance_ended_at = self._agent_speech_started_at - 1e-3
            logger.debug(
                "utterance ended at adjusted: %s",
                self._utterance_ended_at,
            )

        self._utterance_started_at = started_at
        self._overlapping = overlapping
        self._speaking = True

        # TODO: @chenghao-mou remove this debug log
        logger.debug(
            "on_start_of_speech: %s, overlapping: %s",
            started_at,
            overlapping,
        )

    def on_end_of_speech(self, ended_at: float, should_ignore: bool = False) -> None:
        # TODO: @chenghao-mou remove this debug log
        logger.debug(
            "on_end_of_speech: %s, should_ignore: %s",
            ended_at,
            should_ignore,
        )
        if should_ignore and self._overlapping:
            # If user speech started within _AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD of agent speech,
            # don't ignore — TTS leading silence can cause the agent speech timestamp
            # to precede actual audible audio, making this look like a backchannel
            # when it's really the user speaking before hearing the agent.
            if (
                self._utterance_started_at is not None
                and self._agent_speech_started_at is not None
                and self._utterance_started_at - self._agent_speech_started_at
                < _AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD
            ):
                logger.debug(
                    "ignoring should_ignore=True: user speech started %.3fs after agent speech "
                    "(within grace period of %.3fs)",
                    self._utterance_started_at - self._agent_speech_started_at,
                    _AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD,
                )
            else:
                # skip update because it might be a backchannel
                self._overlapping = False
                self._speaking = False
                self._utterance_started_at = None
                self._utterance_ended_at = None
                return

        if (
            self._overlapping
            or (
                self._agent_speech_started_at is not None
                and self._agent_speech_ended_at is None
            )
        ):  # this is an interruption (agent is still speaking)
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
                    "min endpointing delay updated: %s -> %s",
                    prev_val,
                    self.min_delay,
                    extra={
                        "reason": "immediate interruption",
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
                    "max endpointing delay updated: %s -> %s",
                    prev_val,
                    self.max_delay,
                    extra={
                        "reason": "EOT (interruption)",
                        "pause": pause,
                        "max_delay": self.max_delay,
                        "min_delay": self.min_delay,
                        "between_utterance_delay": self.between_utterance_delay,
                        "between_turn_delay": self.between_turn_delay,
                    },
                )

        else:  # this is a normal end of speech
            if (pause := self.between_turn_delay) > 0:
                prev_val = self.max_delay
                self._turn_pause.apply(1.0, pause)
                logger.debug(
                    "max endpointing delay updated due to pause: %s -> %s",
                    prev_val,
                    self.max_delay,
                    extra={
                        "reason": "new turn (case 3)",
                        "pause": pause,
                        "max_delay": self.max_delay,
                        "min_delay": self.min_delay,
                    },
                )
            elif (
                (pause := self.between_utterance_delay) > 0
                and self._agent_speech_ended_at is None
                and self._agent_speech_started_at is None
            ):
                prev_val = self.min_delay
                self._utterance_pause.apply(1.0, pause)
                logger.debug(
                    "min endpointing delay updated: %s -> %s",
                    prev_val,
                    self.min_delay,
                    extra={
                        "reason": "pause between utterances (case 1)",
                        "pause": pause,
                        "max_delay": self.max_delay,
                        "min_delay": self.min_delay,
                    },
                )

        self._utterance_ended_at = ended_at
        self._agent_speech_started_at = None
        self._agent_speech_ended_at = None
        self._speaking = False
        self._overlapping = False

    def update_options(
        self,
        *,
        min_delay: NotGivenOr[float] = NOT_GIVEN,
        max_delay: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(min_delay):
            self._min_delay = min_delay
            self._utterance_pause.reset(initial=self._min_delay, min_val=self._min_delay)
            self._turn_pause.reset(min_val=self._min_delay)

        if is_given(max_delay):
            self._max_delay = max_delay
            self._turn_pause.reset(initial=self._max_delay, max_val=self._max_delay)
            self._utterance_pause.reset(max_val=self._max_delay)


def create_endpointing(options: EndpointingOptions) -> BaseEndpointing:
    match options["mode"]:
        case "dynamic":
            return DynamicEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
            )
        case "fixed":
            return BaseEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
            )
