from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .audio_recognition import TurnDetectionMode


class EndpointingConfig(BaseModel):
    """
    `EndpointingConfig` is the configuration for endpointing.

    Args:
        min_delay (float, optional): Minimum time-in-seconds since the
            last detected speech before the agent declares the user’s turn
            complete. In VAD mode this effectively behaves like
            max(VAD silence, min_delay); in STT mode it is
            applied after the STT end-of-speech signal, so it can be
            additive with the STT provider’s endpointing delay. Defaults to
            ``0.5`` s.
        max_delay (float, optional): Maximum time-in-seconds the agent
            will wait before terminating the turn. Defaults to ``3.0`` s.
    """

    min_delay: NotGivenOr[float] = 0.5
    max_delay: NotGivenOr[float] = 3.0


class InterruptionConfig(BaseModel):
    """
    `InterruptionConfig` is the configuration for interruption handling.

    Args:
        mode (Literal["adaptive", "vad"] | False, optional): Interruption handling strategy.
            Defaults to ``NOT_GIVEN``.
        discard_audio_if_uninterruptible (bool): When ``True``, buffered
            audio is dropped while the agent is speaking and cannot be
            interrupted. Default ``True``.
        min_duration (float): Minimum speech length (s) to
            register as an interruption. Default ``0.5`` s.
        min_words (int): Minimum number of words to consider
            an interruption, only used if stt enabled. Default ``0``.
        false_interruption_timeout (float, optional): If set, emit an
            `agent_false_interruption` event after this amount of time if
            the user is silent and no user transcript is detected after
            the interruption. Set to ``None`` to disable. Default ``2.0`` s.
        resume_false_interruption (bool): Whether to resume the false interruption
            after the false_interruption_timeout. Default ``True``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: NotGivenOr[Literal["adaptive", "vad", False]] = NOT_GIVEN
    discard_audio_if_uninterruptible: bool = True
    # Thresholds
    min_duration: float = 0.5
    min_words: int = 0
    # False interruption
    resume_false_interruption: bool = True
    false_interruption_timeout: float | None = 2.0


class TurnHandlingConfig(BaseModel):
    """
    `TurnHandlingConfig` is the configuration for the turn handling system.

    It is used to configure the turn taking behavior of the session.

    Args:
        turn_detection (TurnDetectionMode, optional): Strategy for deciding
            when the user has finished speaking.

            * ``"stt"`` – rely on speech-to-text end-of-utterance cues
            * ``"vad"`` – rely on Voice Activity Detection start/stop cues
            * ``"realtime_llm"`` – use server-side detection from a
              realtime LLM
            * ``"manual"`` – caller controls turn boundaries explicitly
            * ``_TurnDetector`` instance – plug-in custom detector

            If *NOT_GIVEN*, the session chooses the best available mode in
            priority order ``realtime_llm → vad → stt → manual``; it
            automatically falls back if the necessary model is missing.
        endpointing (EndpointingConfig): Configuration for endpointing.
        interruption (InterruptionConfig): Configuration for interruption handling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_detection: Annotated[NotGivenOr[TurnDetectionMode | None], SkipValidation()] = NOT_GIVEN
    endpointing: EndpointingConfig = Field(default_factory=EndpointingConfig)
    interruption: InterruptionConfig = Field(default_factory=InterruptionConfig)

    @classmethod
    def migrate(
        cls,
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        discard_audio_if_uninterruptible: NotGivenOr[bool] = NOT_GIVEN,
        min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
        min_interruption_words: NotGivenOr[int] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        resume_false_interruption: NotGivenOr[bool] = NOT_GIVEN,
        agent_false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> "TurnHandlingConfig":
        """
        Migrate the turn handling config from the deprecated parameters to the new parameters.
        """
        if is_given(agent_false_interruption_timeout):
            false_interruption_timeout = agent_false_interruption_timeout

        interruption_mode: NotGivenOr[Literal["adaptive", "vad", False]] = NOT_GIVEN
        if allow_interruptions is False:
            interruption_mode = False

        endpointing_kwargs = {}
        # allow not given values for agent to inherit from session
        # AgentSession will always have a value while Agent may not have a value
        endpointing_kwargs["min_delay"] = min_endpointing_delay
        endpointing_kwargs["max_delay"] = max_endpointing_delay

        interruption_kwargs: dict[str, Any] = {}
        if is_given(interruption_mode):
            interruption_kwargs["mode"] = interruption_mode
        if is_given(discard_audio_if_uninterruptible):
            interruption_kwargs["discard_audio_if_uninterruptible"] = (
                discard_audio_if_uninterruptible
            )
        if is_given(min_interruption_duration):
            interruption_kwargs["min_duration"] = min_interruption_duration
        if is_given(min_interruption_words):
            interruption_kwargs["min_words"] = min_interruption_words
        if is_given(false_interruption_timeout):
            interruption_kwargs["false_interruption_timeout"] = false_interruption_timeout
        if is_given(resume_false_interruption):
            interruption_kwargs["resume_false_interruption"] = resume_false_interruption

        kwargs: dict[str, Any] = {}
        if endpointing_kwargs:
            kwargs["endpointing"] = EndpointingConfig(**endpointing_kwargs)
        if interruption_kwargs:
            kwargs["interruption"] = InterruptionConfig(**interruption_kwargs)

        if is_given(turn_detection):
            kwargs["turn_detection"] = turn_detection

        return cls(**kwargs)
