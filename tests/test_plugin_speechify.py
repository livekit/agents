from __future__ import annotations

import asyncio
import base64

import httpx2 as httpx
import pytest
from speechify.types.error_detail import ErrorDetail
from speechify.types.nested_chunk import NestedChunk
from speechify.types.speech_stream_event import (
    SpeechStreamEvent_SpeechChunk,
    SpeechStreamEvent_SpeechDone,
    SpeechStreamEvent_SpeechError,
)

from livekit.agents import APIConnectOptions, APIStatusError, tts
from livekit.agents.types import NOT_GIVEN, USERDATA_TIMED_TRANSCRIPT
from livekit.plugins.speechify import tts as sfy_tts

pytestmark = pytest.mark.unit

_PCM_1S = b"\x00\x7f" * (24000 * 2)  # 0.5s of 24 kHz mono PCM silence


class _FakeAudio:
    def __init__(self) -> None:
        self.stream_calls: list[dict] = []
        self.speech_calls: list[dict] = []
        self.streams_read_to_end = 0

    async def stream_with_timestamps(self, **kwargs: object) -> object:
        self.stream_calls.append(kwargs)
        yield SpeechStreamEvent_SpeechChunk(
            type="speech.chunk",
            audio=base64.b64encode(_PCM_1S).decode("ascii"),
            speech_marks=[
                NestedChunk(
                    type="word", value="Hello", start=0, end=5, start_time=0, end_time=320.0
                )
            ],
        )
        yield SpeechStreamEvent_SpeechChunk(
            type="speech.chunk",
            audio=None,
            speech_marks=[
                NestedChunk(
                    type="word", value="world", start=6, end=11, start_time=340.0, end_time=700.0
                )
            ],
        )
        yield SpeechStreamEvent_SpeechDone(
            type="speech.done", billable_characters_count=11, audio_duration_ms=700
        )
        # only reached when the caller keeps reading after speech.done
        self.streams_read_to_end += 1

    async def speech(self, **kwargs: object) -> _FakeSpeechResponse:
        self.speech_calls.append(kwargs)
        return _FakeSpeechResponse()

    async def stream(self, **kwargs: object) -> object:
        raise AssertionError("raw /v1/audio/stream should not be called")


class _FakeSpeechResponse:
    audio_data: str = base64.b64encode(_PCM_1S).decode("ascii")
    speech_marks = type("Marks", (), {"chunks": []})


class _FakeVoices:
    async def list(self) -> list[object]:
        return []


class _ErrorAudio(_FakeAudio):
    async def stream_with_timestamps(self, **kwargs: object) -> object:
        self.stream_calls.append(kwargs)
        yield SpeechStreamEvent_SpeechError(
            type="speech.error",
            error=ErrorDetail(code="speech_failed", message="boom"),
            request_id="rid-1",
        )


class _FakeClient:
    def __init__(self) -> None:
        self.audio = _FakeAudio()
        self.voices = _FakeVoices()


_NO_RETRY = APIConnectOptions(max_retry=0)


def _make_tts(model: str, client: _FakeClient) -> sfy_tts.TTS:
    return sfy_tts.TTS(model=model, client=client)  # type: ignore[arg-type]


async def _collect(stream: tts.SynthesizeStream) -> list[tts.SynthesizedAudio]:
    frames: list[tts.SynthesizedAudio] = []
    async for frame in stream:
        frames.append(frame)
    return frames


async def test_stream_with_timestamps() -> None:
    client = _FakeClient()
    stream = _make_tts("simba-3.2", client).stream()
    stream.push_text("Hello world.")
    stream.end_input()

    frames = await _collect(stream)

    # the streaming-native model must be routed to the SSE endpoint
    assert len(client.audio.stream_calls) == 1
    kwargs = client.audio.stream_calls[0]
    assert kwargs["input"] == "Hello world."
    assert kwargs["voice_id"] == sfy_tts.DEFAULT_VOICE_ID
    assert kwargs["output_format"] == "pcm_24000"
    assert kwargs.get("model") == "simba-3.2"
    assert client.audio.speech_calls == []

    # 0.5s of PCM audio was pushed and its word marks surfaced
    total = sum(f.frame.duration for f in frames)
    assert total > 0.4
    transcripts = [t for f in frames for t in f.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    words = {str(t): t for t in transcripts}
    assert set(words) == {"Hello", "world"}
    assert words["Hello"].start_time == pytest.approx(0.0)
    assert words["Hello"].end_time == pytest.approx(0.32)
    assert words["world"].start_time == pytest.approx(0.34)

    # the body is read past speech.done, so its connection can go back to the pool
    assert client.audio.streams_read_to_end == 1


async def test_stream_falls_back_for_legacy_model() -> None:
    client = _FakeClient()
    stream = _make_tts("simba-english", client).stream()
    stream.push_text("Hello world.")
    stream.end_input()

    frames = await _collect(stream)

    # legacy models do not serve the SSE route -> per-sentence /v1/audio/speech
    assert client.audio.stream_calls == []
    assert len(client.audio.speech_calls) == 1
    assert sum(f.frame.duration for f in frames) > 0.4


async def test_stream_not_given_model_uses_streaming() -> None:
    client = _FakeClient()
    stream = sfy_tts.TTS(model=NOT_GIVEN, client=client).stream()  # type: ignore[arg-type]
    stream.push_text("Hello world.")
    stream.end_input()

    await _collect(stream)

    # when no model is configured the server defaults to simba-3.0 (streaming-native)
    assert len(client.audio.stream_calls) == 1
    assert "model" not in client.audio.stream_calls[0]


async def test_stream_predicates_speech_error() -> None:
    client = _FakeClient()
    client.audio = _ErrorAudio()
    stream = _make_tts("simba-3.2", client).stream()
    stream.push_text("Hello world.")
    stream.end_input()

    with pytest.raises(APIStatusError, match="boom"):
        await _collect(stream)


async def test_synthesize_streams_with_timestamps() -> None:
    client = _FakeClient()
    frames = [f async for f in _make_tts("simba-3.2", client).synthesize("Hello world.")]

    # synthesize() now streams on the streaming-native models instead of one batch request
    assert len(client.audio.stream_calls) == 1
    assert client.audio.stream_calls[0]["input"] == "Hello world."
    assert client.audio.stream_calls[0]["output_format"] == "pcm_24000"
    assert client.audio.speech_calls == []
    assert client.audio.streams_read_to_end == 1

    assert sum(f.frame.duration for f in frames) > 0.4
    transcripts = [t for f in frames for t in f.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    assert [str(t) for t in transcripts] == ["Hello", "world"]
    assert transcripts[1].start_time == pytest.approx(0.34)


async def test_synthesize_uses_speech_endpoint_for_legacy_model() -> None:
    client = _FakeClient()
    frames = [f async for f in _make_tts("simba-english", client).synthesize("Hello world.")]

    assert client.audio.stream_calls == []
    assert len(client.audio.speech_calls) == 1
    assert client.audio.speech_calls[0]["audio_format"] == "pcm"
    assert sum(f.frame.duration for f in frames) > 0.4


async def test_synthesize_predicates_speech_error() -> None:
    client = _FakeClient()
    client.audio = _ErrorAudio()
    t = _make_tts("simba-3.2", client)

    with pytest.raises(APIStatusError, match="boom"):
        async with t.synthesize("Hello world.", conn_options=_NO_RETRY) as stream:
            async for _ in stream:
                pass


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = httpx.MockTransport(handler)  # type: ignore[arg-type]
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


async def test_prewarm_opens_connection_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    _mock_http(monkeypatch, handler)
    t = sfy_tts.TTS(api_key="sk-test", base_url="https://api.example.test/")
    t.prewarm()
    t.prewarm()  # a second call while the first is in flight is a no-op
    assert t._prewarm_task is not None
    await t._prewarm_task

    assert [(r.method, str(r.url)) for r in requests] == [
        ("GET", "https://api.example.test/health")
    ]
    assert "authorization" not in requests[0].headers
    assert requests[0].headers[sfy_tts.CALLER_HEADER] == "livekit"
    await t.aclose()


async def test_aclose_cancels_pending_prewarm(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _mock_http(monkeypatch, handler)
    t = sfy_tts.TTS(api_key="sk-test")
    t.prewarm()
    task = t._prewarm_task
    assert task is not None

    await asyncio.wait_for(t.aclose(), timeout=1)
    assert task.cancelled()


def test_prewarm_is_noop_with_injected_client() -> None:
    t = _make_tts("simba-3.2", _FakeClient())
    t.prewarm()
    assert t._prewarm_task is None


def test_supports_streaming_marks() -> None:
    assert sfy_tts._supports_streaming_marks(NOT_GIVEN)
    assert sfy_tts._supports_streaming_marks("simba-3.0")
    assert sfy_tts._supports_streaming_marks("simba-3.2")
    assert not sfy_tts._supports_streaming_marks("simba-english")
    assert not sfy_tts._supports_streaming_marks("simba-multilingual")


def test_marks_to_timed_uses_start_end_time() -> None:
    marks = [
        NestedChunk(type="word", value="a", start=0, end=1, start_time=0.0, end_time=100.0),
        # character-offset-only mark (no times) must be skipped, as in the batch path
        NestedChunk(type="word", value="b", start=5, end=6, start_time=None, end_time=None),
    ]
    timed = sfy_tts._marks_to_timed(marks, offset=0.5)
    assert len(timed) == 1
    assert timed[0] == "a"
    assert timed[0].start_time == pytest.approx(0.5)
    assert timed[0].end_time == pytest.approx(0.6)


def test_warn_voice_model_compat(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", "livekit.plugins.speechify"):
        # curated simba-3.2 voices (id suffix "_32") warn on other models
        sfy_tts._warn_voice_model_compat("dominic_32", "simba-english")
        assert any("dominic_32" in r.message for r in caplog.records)

        caplog.clear()
        # NOT_GIVEN defaults the server to simba-3.0 -> still a mismatch
        sfy_tts._warn_voice_model_compat("dominic_32", NOT_GIVEN)
        assert any("dominic_32" in r.message for r in caplog.records)

        caplog.clear()
        # the supported pairing is silent
        sfy_tts._warn_voice_model_compat("dominic_32", "simba-3.2")
        assert not caplog.records

        caplog.clear()
        # non "_32" voices never warn regardless of model
        sfy_tts._warn_voice_model_compat("henry", "simba-english")
        sfy_tts._warn_voice_model_compat("henry", "simba-3.2")
        assert not caplog.records
