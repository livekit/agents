from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc


@dataclass
class TranslationCapabilities:
    """What varies between translation providers."""

    input_transcription: bool = False
    """Whether the model transcribes the speech it hears, in the language it was spoken."""
    output_transcription: bool = False
    """Whether the model transcribes the translation it speaks."""
    turn_reporting: bool = False
    """Whether the model reports where the speaker's turns start and end.

    About the speech it hears. For the speech it speaks, see ``output_segmentation``.
    """
    output_segmentation: bool = False
    """Whether the model marks where each translation it speaks ends.

    When it does, the session ends a playout segment on that mark alone. When it does not,
    the session falls back to a silence timer, which cannot tell a pause inside a sentence
    from the end of one.
    """
    retargetable: bool = False
    """Whether the target language can be changed on a running session."""


@dataclass
class TranslatedAudioFrame:
    """One frame of the translation the model speaks."""

    frame: rtc.AudioFrame
    language: str | None = None
    """The language spoken in this frame, for a provider that stamps its audio."""


@dataclass
class InputTranscript:
    """What the model heard, in the language it was spoken."""

    text: str
    is_final: bool
    language: str | None = None
    speaker_id: str | None = None


@dataclass
class OutputTranscript:
    """What the model said, in the language it translates into."""

    text: str
    is_final: bool
    language: str | None = None


@dataclass
class InputSpeechStarted:
    """The speaker began a turn, as the model heard it."""


@dataclass
class InputSpeechStopped:
    """The speaker ended a turn, as the model heard it."""


@dataclass
class OutputSpeechStopped:
    """The model finished speaking a translation; its audio up to here is one segment.

    Yielded by :attr:`TranslationModelSession.audio_stream` rather than emitted as an event,
    so it cannot overtake audio the model has already queued ahead of it.
    """


class TranslationModelError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["translation_model_error"] = "translation_model_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TranslationEventTypes = Literal[
    "input_speech_started",  # the speaker's turns, as the model reports them
    "input_speech_stopped",
    "input_transcript",  # the model's transcript of what it heard
    "output_transcript",  # the model's transcript of what it said
    "error",
]

TEvent = TypeVar("TEvent")


class TranslationModel(ABC):
    """A model that hears speech in one language and speaks it in another.

    Its own primitive, beside :class:`~livekit.agents.llm.LLM`,
    :class:`~livekit.agents.llm.RealtimeModel` and :class:`~livekit.agents.llm.DuplexModel`: a
    translator holds no conversation, so it takes no instructions, carries no chat context,
    calls no tools and never decides to reply. What it has instead is a language it translates
    into. Run one with
    :class:`~livekit.agents.beta.translation.TranslationSession`.
    """

    def __init__(
        self,
        *,
        capabilities: TranslationCapabilities,
        target_language: str,
        source_language: str | None = None,
    ) -> None:
        """
        Args:
            capabilities: What this provider can do.
            target_language: The language the model speaks its translation in.
            source_language: The language the model expects to hear, or ``None`` when it
                detects the language itself.
        """
        self._capabilities = capabilities
        self._target_language = target_language
        self._source_language = source_language
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    @property
    def capabilities(self) -> TranslationCapabilities:
        return self._capabilities

    @property
    def label(self) -> str:
        return self._label

    @property
    def target_language(self) -> str:
        """The language the model speaks its translation in."""
        return self._target_language

    @property
    def source_language(self) -> str | None:
        """The language the model expects to hear, or ``None`` when it detects it itself."""
        return self._source_language

    def update_options(self, *, target_language: str) -> None:
        """Point the model at another language.

        Takes effect on the sessions opened after it. To retarget a running session, call
        :meth:`TranslationModelSession.update_options` on the session itself, which a
        provider only serves when ``capabilities.retargetable``.
        """
        self._target_language = target_language

    @abstractmethod
    def session(self) -> TranslationModelSession:
        """Open a session; the caller pushes audio into it and reads its translation out."""

    @abstractmethod
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> TranslationModel:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class TranslationModelSession(
    ABC, rtc.EventEmitter[TranslationEventTypes | TEvent], Generic[TEvent]
):
    """One run of a :class:`TranslationModel`: audio in, translated audio out.

    The speech the model hears is pushed in frame by frame with :meth:`push_audio`; the
    translation it speaks is read out of :attr:`audio_stream`, which also carries an
    :class:`OutputSpeechStopped` where each spoken translation ends. What it heard and what it
    said arrive as ``input_transcript`` and ``output_transcript`` events, and the speaker's
    turns as ``input_speech_started`` and ``input_speech_stopped``.
    """

    def __init__(self, translation_model: TranslationModel) -> None:
        super().__init__()
        self._translation_model = translation_model
        self._closing = False

    @property
    def translation_model(self) -> TranslationModel:
        return self._translation_model

    @property
    def capabilities(self) -> TranslationCapabilities:
        return self._translation_model.capabilities

    @property
    @abstractmethod
    def audio_stream(self) -> AsyncIterable[TranslatedAudioFrame | OutputSpeechStopped]:
        """The translation the model speaks, and where each one ends.

        A model whose ``capabilities.output_segmentation`` is set yields an
        :class:`OutputSpeechStopped` after the last frame of each translation. Because it
        travels in the stream, it is always read after the audio it closes.
        """

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Hand the model one frame of the speech to translate."""

    def clear_audio(self) -> None:
        """Drop the translation the model has not spoken yet.

        A no-op for a provider that cannot discard what it has already produced.
        """

    def update_options(self, *, target_language: str) -> None:
        """Point this running session at another language.

        Raises:
            RuntimeError: If the provider fixes the target language for the session's life.
        """
        raise RuntimeError(
            f"{self._translation_model.label} translates into a language fixed for the "
            "session's life; start a new session to translate into another"
        )

    @abstractmethod
    async def aclose(self) -> None: ...
