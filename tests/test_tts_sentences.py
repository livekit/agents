"""``tts.StreamAdapter``'s sentences and ``SentenceStreamPacer``: the core splits, paces and
synthesizes; a custom ``sentence_tokenizer`` and a plugin's own sentence stream stay the host's."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectOptions, tokenize, tts

pytestmark = pytest.mark.unit

_CONN = APIConnectOptions()
_RATE = 24000


class _BatchTTS(tts.TTS):
    """Synthesizes each text as ``seconds`` of silence and records it."""

    def __init__(self, seconds: float = 0.1) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=_RATE, num_channels=1
        )
        self.seconds = seconds
        self.texts: list[str] = []

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        self.texts.append(chunk.text)
        return _Silence(tts=self, input_text=chunk.text, conn_options=conn_options)


class _Silence(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, _BatchTTS)
        output_emitter.initialize(
            request_id="r", sample_rate=_RATE, num_channels=1, mime_type="audio/pcm"
        )
        output_emitter.push(b"\x00\x00" * int(_RATE * self._tts.seconds))


class _BarStream(tokenize.BufferedSentenceStream):
    def __init__(self, owner: _BarTokenizer) -> None:
        super().__init__(tokenizer=self._split, min_token_len=1, min_ctx_len=1)
        self._owner = owner

    def _split(self, text: str) -> list[str]:
        return [part for part in text.split("|") if part]

    async def aclose(self) -> None:
        self._owner.closed += 1
        await super().aclose()


class _BarTokenizer(tokenize.SentenceTokenizer):
    """Splits on ``|``: nothing a default tokenizer would do."""

    def __init__(self) -> None:
        self.closed = 0

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [part for part in text.split("|") if part]

    def stream(self, *, language: str | None = None) -> tokenize.SentenceStream:
        return _BarStream(self)


async def _speak(adapter: tts.StreamAdapter, *chunks: str) -> list[tts.SynthesizedAudio]:
    async with asyncio.timeout(10):
        async with adapter.stream() as stream:
            for chunk in chunks:
                stream.push_text(chunk)
            stream.end_input()
            return [audio async for audio in stream]


# Catches: a custom sentence_tokenizer in `options` ignored in favor of the default split, or its
# stream left open after the synthesis.
async def test_a_custom_sentence_tokenizer_splits_the_adapters_text() -> None:
    provider, tokenizer = _BatchTTS(), _BarTokenizer()
    provider.options = tts.TTSOptions(sentence_tokenizer=tokenizer, pacing=None)
    adapter = tts.StreamAdapter(tts=provider)
    try:
        frames = await _speak(adapter, "one two|three", " four|five")
        assert provider.texts == ["one two", "three four", "five"]
        assert frames[-1].is_final and not any(f.is_final for f in frames[:-1])
        assert tokenizer.closed == 1
    finally:
        await adapter.aclose()
