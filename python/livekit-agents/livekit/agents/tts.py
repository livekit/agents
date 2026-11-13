from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import rtc

from . import _core, tokenize, utils
from ._generated import facades_tts as _facades, public_tts as _public
from ._generated.public_tts import *  # noqa: F403
from .types import (
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    NotGivenOr,
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
    """Several TTS behind one: a failed one is skipped until it recovers. Each prepares its text
    by its own options; ``options`` are the serving TTS's."""

    def prewarm(self) -> None:
        self._members[0].prewarm()

    def update_options(
        self,
        *,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Change how every TTS of the chain prepares its text."""
        for member in self._members:
            member.update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)

    def _core_options(self) -> _core.TtsOptionsCell:
        return self._active._core_options()


class StreamAdapter(_facades.StreamAdapter):
    """A non-streaming TTS as a streaming one: each chunk of the text, as the wrapped TTS's
    options split it, is synthesized on its own."""

    class Markup(TTS.Markup):
        # a pass-through speaks whatever dialect it wraps
        def _provider_key(self) -> str:
            assert isinstance(self._tts, StreamAdapter)
            return self._tts._tts.markup._provider_key()

    def _core_options(self) -> _core.TtsOptionsCell:
        # the wrapped TTS's: the adapter prepares its text as that TTS would, and
        # update_options changes them
        return self._tts._core_options()

    @property
    def model(self) -> str:
        return self._tts.model

    @property
    def provider(self) -> str:
        return self._tts.provider

    def prewarm(self) -> None:
        self._tts.prewarm()
