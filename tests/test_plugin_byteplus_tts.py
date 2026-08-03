from __future__ import annotations

import base64
import io
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APIStatusError, tts, utils
from livekit.agents.tokenize.tokenizer import TokenData
from livekit.agents.voice.io import TimedString
from livekit.plugins import byteplus
from livekit.plugins.byteplus import TTS, AIGCMetadata, TTSUsage, tts as byteplus_tts
from livekit.plugins.byteplus.tts import SynthesizeStream, _request_and_emit_audio

pytestmark = pytest.mark.plugin("byteplus")


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        raise AssertionError("response content must be consumed with iter_any()")

    def iter_any(self) -> AsyncIterator[bytes]:
        return self._iter_chunks()

    async def _iter_chunks(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    headers = {"X-Tt-Logid": "provider-request-id"}

    def __init__(self, chunks: list[bytes], *, status: int = 200) -> None:
        self.content = _FakeContent(chunks)
        self.status = status

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def text(self) -> str:
        return ""


class _FakeSession:
    def __init__(self, responses: list[list[bytes]], *, status: int = 200) -> None:
        self._responses = iter(responses)
        self._status = status

    def post(self, _url: str, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(next(self._responses), status=self._status)


class _FakeEmitter:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.timed_words: list[TimedString] = []

    def push(self, data: bytes) -> None:
        self.audio.extend(data)

    def push_timed_transcript(self, words: list[TimedString]) -> None:
        self.timed_words.extend(words)

    def _note_provider_request_id(self, _request_id: str) -> None:
        pass


class _FakeStreamingEmitter(_FakeEmitter):
    def initialize(self, **_kwargs: object) -> None:
        pass

    def start_segment(self, *, segment_id: str) -> None:
        pass

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        pass


class _FailingSentenceStream:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self._sent = False

    def push_text(self, _text: str) -> None:
        pass

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        pass

    async def aclose(self) -> None:
        self._order.append("sentence_stream.aclose")

    def __aiter__(self) -> _FailingSentenceStream:
        return self

    async def __anext__(self) -> TokenData:
        if self._sent:
            await utils.aio.sleep(3600)
            raise StopAsyncIteration
        self._sent = True
        return TokenData(token="hello")


class _FailingSentenceTokenizer:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def stream(self) -> _FailingSentenceStream:
        return _FailingSentenceStream(self._order)


def _response_events(pcm: bytes, *, include_timings: bool) -> list[bytes]:
    event: dict[str, Any] = {
        "code": 0,
        "data": base64.b64encode(pcm).decode("ascii"),
    }
    if include_timings:
        event["sentence"] = {
            "words": [
                {
                    "word": "hello",
                    "startTime": 0.1,
                    "endTime": 0.4,
                    "confidence": 1.0,
                }
            ]
        }
    return [
        json.dumps(event).encode("utf-8") + b'{"code": 20000000, "message": "ok"}',
    ]


def _encode_mp3(*, sample_rate: int, duration: float) -> bytes:
    import av

    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp3")
    stream = container.add_stream("libmp3lame", rate=sample_rate)
    frame = av.AudioFrame(
        format="s16",
        layout="mono",
        samples=int(sample_rate * duration),
    )
    frame.sample_rate = sample_rate
    frame.planes[0].update(b"\0\0" * frame.samples)
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def test_public_option_and_event_types_are_exported() -> None:
    public_types = {AIGCMetadata.__name__, TTSUsage.__name__}
    assert public_types <= set(byteplus.__all__)
    assert public_types.isdisjoint(byteplus.__pdoc__)


@pytest.mark.parametrize("invalid_value", [False, True, 0.0, 100.0])
def test_parenthesis_filter_requires_an_exact_integer_choice(invalid_value: object) -> None:
    with pytest.raises(ValueError, match="max_length_to_filter_parenthesis"):
        TTS(
            api_key="test-key",
            max_length_to_filter_parenthesis=cast(Any, invalid_value),
        )

    provider = TTS(api_key="test-key")
    with pytest.raises(ValueError, match="max_length_to_filter_parenthesis"):
        provider.update_options(
            max_length_to_filter_parenthesis=cast(Any, invalid_value),
        )


@pytest.mark.parametrize("valid_value", [0, 100])
def test_parenthesis_filter_accepts_documented_integer_choices(valid_value: int) -> None:
    provider = TTS(
        api_key="test-key",
        max_length_to_filter_parenthesis=cast(Any, valid_value),
    )
    assert provider._opts.max_length_to_filter_parenthesis == valid_value


@pytest.mark.parametrize("first_request_has_timings", [False, True])
async def test_sentence_timing_offset_uses_emitted_audio_duration(
    first_request_has_timings: bool,
) -> None:
    sample_rate = 24000
    one_second_pcm = b"\0\0" * sample_rate
    session = _FakeSession(
        [
            _response_events(one_second_pcm, include_timings=first_request_has_timings),
            _response_events(one_second_pcm, include_timings=True),
        ]
    )
    emitter = _FakeEmitter()
    provider = TTS(api_key="test-key")
    conn_options = APIConnectOptions(max_retry=0, timeout=1.0)

    transcript_offset = await _request_and_emit_audio(
        provider=provider,
        session=cast(aiohttp.ClientSession, session),
        opts=provider._opts,
        text="first",
        output_emitter=cast(tts.AudioEmitter, emitter),
        conn_options=conn_options,
        request_id="first-request",
        transcript_offset=0.0,
    )
    assert transcript_offset == pytest.approx(1.0)

    request_duration = await _request_and_emit_audio(
        provider=provider,
        session=cast(aiohttp.ClientSession, session),
        opts=provider._opts,
        text="second",
        output_emitter=cast(tts.AudioEmitter, emitter),
        conn_options=conn_options,
        request_id="second-request",
        transcript_offset=transcript_offset,
    )

    second_word = emitter.timed_words[-1]
    assert second_word.start_time == pytest.approx(1.1)
    assert second_word.end_time == pytest.approx(1.4)
    assert request_duration == pytest.approx(1.0)


async def test_compressed_request_duration_uses_decoded_audio() -> None:
    sample_rate = 24000
    compressed_audio = _encode_mp3(sample_rate=sample_rate, duration=0.5)
    session = _FakeSession([_response_events(compressed_audio, include_timings=False)])
    emitter = _FakeEmitter()
    provider = TTS(
        api_key="test-key",
        audio_format="mp3",
        sample_rate=sample_rate,
    )

    request_duration = await _request_and_emit_audio(
        provider=provider,
        session=cast(aiohttp.ClientSession, session),
        opts=provider._opts,
        text="compressed",
        output_emitter=cast(tts.AudioEmitter, emitter),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        request_id="compressed-request",
        transcript_offset=0.0,
        decode_compressed_audio=True,
    )

    assert request_duration == pytest.approx(0.5, abs=0.05)
    assert emitter.audio


async def test_stream_cancels_input_task_before_closing_sentence_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    provider = TTS(
        api_key="test-key",
        http_session=cast(
            aiohttp.ClientSession,
            _FakeSession([[b""]], status=500),
        ),
        tokenizer=cast(Any, _FailingSentenceTokenizer(order)),
    )
    stream = object.__new__(SynthesizeStream)
    stream._tts = provider
    stream._opts = provider._opts
    stream._conn_options = APIConnectOptions(max_retry=0, timeout=1.0)
    stream._input_ch = utils.aio.Chan[str | SynthesizeStream._FlushSentinel]()
    stream._started_time = 0.0

    async def fake_gracefully_cancel(*tasks: object) -> None:
        order.append("gracefully_cancel")
        await utils.aio.cancel_and_wait(*cast(Any, tasks))

    monkeypatch.setattr(byteplus_tts.utils.aio, "gracefully_cancel", fake_gracefully_cancel)

    with pytest.raises(APIStatusError):
        await stream._run(cast(tts.AudioEmitter, _FakeStreamingEmitter()))

    assert order == ["gracefully_cancel", "sentence_stream.aclose"]
