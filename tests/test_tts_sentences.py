"""``tts.StreamAdapter``'s sentences and ``SentenceStreamPacer``: the core splits, paces and
synthesizes; a custom ``sentence_tokenizer`` and a plugin's own sentence stream stay the host's."""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents import APIConnectOptions, tokenize, tts, utils

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

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = _CONN
    ) -> tts.ChunkedStream:
        self.texts.append(text)
        return _Silence(tts=self, input_text=text, conn_options=conn_options)


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


# Catches: a custom sentence_tokenizer ignored in favor of the default split, or its stream left
# open after the synthesis.
async def test_a_custom_sentence_tokenizer_splits_the_adapters_text() -> None:
    provider, tokenizer = _BatchTTS(), _BarTokenizer()
    adapter = tts.StreamAdapter(tts=provider, sentence_tokenizer=tokenizer)
    try:
        frames = await _speak(adapter, "one two|three", " four|five")
        assert provider.texts == ["one two", "three four", "five"]
        assert frames[-1].is_final and not any(f.is_final for f in frames[:-1])
        assert tokenizer.closed == 1
    finally:
        await adapter.aclose()


# Catches: the default split losing blingfire's sentence boundaries when the core runs it.
async def test_the_default_split_is_blingfires() -> None:
    provider = _BatchTTS()
    adapter = tts.StreamAdapter(tts=provider)
    try:
        await _speak(adapter, "The first sentence is long enough. ", "And here is the second one.")
        assert provider.texts == [
            "The first sentence is long enough.",
            "And here is the second one.",
        ]
    finally:
        await adapter.aclose()


# Catches: pacing that sends every sentence at once, or loses one held back.
async def test_text_pacing_sends_the_first_sentence_then_batches_the_rest() -> None:
    provider = _BatchTTS(seconds=0.1)
    adapter = tts.StreamAdapter(
        tts=provider,
        sentence_tokenizer=_BarTokenizer(),
        text_pacing=tts.SentenceStreamPacer(min_remaining_audio=5.0, max_text_length=300),
    )
    try:
        await _speak(adapter, "one|two|three|", "four|")
        assert provider.texts[0] == "one"
        assert " ".join(provider.texts) == "one two three four"
        assert len(provider.texts) < 4
    finally:
        await adapter.aclose()


def _emitter() -> tuple[tts.AudioEmitter, utils.aio.Chan[tts.SynthesizedAudio]]:
    ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test", dst_ch=ch)
    emitter.initialize(
        request_id="r", sample_rate=_RATE, num_channels=1, mime_type="audio/pcm", stream=True
    )
    emitter.start_segment(segment_id="s")
    return emitter, ch


# Catches: a plugin's wrapped stream not paced by its emitter's audio, or not closed with it.
async def test_wrap_paces_a_plugins_sentence_stream_by_its_emitters_audio() -> None:
    emitter, _ch = _emitter()
    tokenizer = _BarTokenizer()
    paced = tts.SentenceStreamPacer(min_remaining_audio=10.0).wrap(tokenizer.stream(), emitter)
    try:
        async with asyncio.timeout(10):
            paced.push_text("one|two|three|")
            started = time.monotonic()
            assert (await paced.__anext__()).token == "one"
            assert time.monotonic() - started < 0.5
            # held until the audio pushed for the first sentence stops growing
            emitter.push(b"\x00\x00" * _RATE)
            paced.end_input()
            assert (await paced.__anext__()).token == "two three"
            assert time.monotonic() - started >= 0.1
            with pytest.raises(StopAsyncIteration):
                await paced.__anext__()
    finally:
        await paced.aclose()
        await emitter.aclose()
    assert tokenizer.closed == 1


# Catches: aclose() leaving a pending read of a wrapped stream hanging.
async def test_closing_a_wrapped_stream_ends_its_pending_read() -> None:
    emitter, _ch = _emitter()
    paced = tts.SentenceStreamPacer().wrap(_BarTokenizer().stream(), emitter)
    pending = asyncio.ensure_future(paced.__anext__())
    await asyncio.sleep(0.05)
    try:
        async with asyncio.timeout(5):
            await paced.aclose()
            with pytest.raises(StopAsyncIteration):
                await pending
    finally:
        await emitter.aclose()
