from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import rtc

from . import utils
from ._generated import facades_tts as _facades, public_tts as _public
from ._generated.public_tts import *  # noqa: F403
from .types import (
    USERDATA_TIMED_TRANSCRIPT,
    TranscriptChunk,
)

if TYPE_CHECKING:
    from ._generated.usage import TokenUsage

__all__ = [
    "StreamAdapter",
    "FallbackAdapter",
]
__all__ += _public.__all__


class _EmitterOutput:
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
    """A non-streaming TTS as a streaming one: each chunk of the text, as the wrapped TTS's
    options split it, is synthesized on its own."""

    class Markup(TTS.Markup):
        # a pass-through speaks whatever dialect it wraps
        def _provider_key(self) -> str:
            assert isinstance(self._tts, StreamAdapter)
            return self._tts._tts.markup._provider_key()

    @property
    def options(self) -> TTSOptions:
        """The wrapped TTS's: the adapter prepares its text as that TTS would."""
        return self._tts.options

    @options.setter
    def options(self, value: TTSOptions) -> None:
        self._tts.options = value

    @property
    def model(self) -> str:
        return self._tts.model

    @property
    def provider(self) -> str:
        return self._tts.provider

    def prewarm(self) -> None:
        self._tts.prewarm()
