"""Text to speech: the base classes a provider subclasses (generated from the core's
declarations), the expressive markup helpers over the core's per-provider tables
(``_core.Markup``), and the adapters, whose engines are the core's (``_core.FallbackTts``,
``_core.TtsStreamAdapter``, ``_core._core_PacedSentences``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import rtc

from . import utils
from ._generated import facades_tts as _facades, public_tts as _public
from ._generated.public_tts import *  # noqa: F403
from .tokenize import SentenceStream
from .types import (
    DEFAULT_API_CONNECT_OPTIONS,
    USERDATA_TIMED_TRANSCRIPT,
    APIConnectOptions,
    TranscriptChunk,
)

if TYPE_CHECKING:
    from ._generated.usage import TokenUsage

__all__ = [
    "StreamAdapter",
    "FallbackAdapter",
    "SentenceStreamPacer",
]
__all__ += _public.__all__


class _EmitterOutput:
    """Where the native emitter's output goes: ``SynthesizedAudio`` into the synthesis channel,
    and the provider's request ids onto its current span."""

    def __init__(self, ch: utils.aio.Chan[SynthesizedAudio]) -> None:
        self._ch = ch

    def send(
        self,
        frame: rtc.AudioFrame,
        request_id: str,
        segment_id: str,
        is_final: bool,
        timed_transcript: list[TranscriptChunk],
        usage: TokenUsage | None,
        characters_count: int | None,
        provider_request_ids: list[str],
    ) -> None:
        if self._ch.closed:
            return
        frame.userdata[USERDATA_TIMED_TRANSCRIPT] = timed_transcript
        self._ch.send_nowait(
            SynthesizedAudio(
                frame=frame,
                request_id=request_id,
                segment_id=segment_id,
                is_final=is_final,
                usage=usage,
                characters_count=characters_count,
                provider_request_ids=provider_request_ids,
            )
        )

    def closed(self) -> bool:
        return self._ch.closed


class FallbackAdapter(_facades.FallbackAdapter):
    """Several TTS behind one: a failed one is skipped until it recovers."""

    def prewarm(self) -> None:
        self._members[0].prewarm()


class StreamAdapter(_facades.StreamAdapter):
    """A non-streaming TTS as a streaming one: the text is split into sentences, each
    synthesized on its own (``_core.TtsStreamAdapter``)."""

    class Markup(TTS.Markup):
        # a pass-through speaks whatever dialect it wraps
        def _provider_key(self) -> str:
            assert isinstance(self._tts, StreamAdapter)
            return self._tts._tts.markup._provider_key()

    def _core_lowers_markup(self) -> bool:
        # its sentences are lowered here, unlike a natively streaming plugin's input
        return True

    @property
    def model(self) -> str:
        return self._tts.model

    @property
    def provider(self) -> str:
        return self._tts.provider

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return self._tts.synthesize(text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        self._tts.prewarm()


class SentenceStreamPacer:
    def __init__(self, *, min_remaining_audio: float = 5.0, max_text_length: int = 300) -> None:
        """Paces the text sent to a TTS: sentences are batched and sent when the audio left to
        play runs low, which wastes less on interruptions and gives the TTS more context.

        The first sentence goes at once. Each later batch waits until the synthesized audio
        stopped growing and at most ``min_remaining_audio`` seconds of it are left to play
        (measured from the first audio pushed), then joins whole sentences with a space until
        it reaches ``max_text_length`` characters.

        Args:
            min_remaining_audio: The audio left, in seconds, below which the next batch goes.
            max_text_length: The longest text sent at once.
        """
        self.min_remaining_audio = min_remaining_audio
        self.max_text_length = max_text_length

    def wrap(self, sent_stream: SentenceStream, audio_emitter: AudioEmitter) -> StreamPacerWrapper:
        """``sent_stream``'s sentences, paced by the audio ``audio_emitter`` pushed. The core
        reads ``sent_stream`` and runs the pacing (``_core._core_PacedSentences``); text pushed
        to the wrapper goes to ``sent_stream``, and closing the wrapper closes it."""
        return StreamPacerWrapper(sent_stream, audio_emitter, pacer=self)
