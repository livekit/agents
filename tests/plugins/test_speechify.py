"""Speechify TTS: SSE streaming with word timestamps and speech errors."""

from __future__ import annotations

import base64

import pytest
from speechify.types.error_detail import ErrorDetail
from speechify.types.nested_chunk import NestedChunk
from speechify.types.speech_stream_event import (
    SpeechStreamEvent_SpeechChunk,
    SpeechStreamEvent_SpeechDone,
    SpeechStreamEvent_SpeechError,
)

from livekit.agents import APIStatusError, tts
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, APIConnectOptions
from livekit.plugins.speechify import tts as sfy_tts

_PCM_1S = b"\x00\x7f" * (24000 * 2)  # 0.5s of 24 kHz mono PCM silence


class _FakeAudio:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, object]] = []
        self.speech_calls: list[dict[str, object]] = []
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
    billable_characters_count = 12
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


def _make_tts(model: str, client: _FakeClient) -> sfy_tts.TTS:
    # the stand-in implements only the SDK calls the plugin makes
    return sfy_tts.TTS(model=model, client=client)  # type: ignore[arg-type]


async def _collect(stream: tts.SynthesizeStream) -> list[tts.SynthesizedAudio]:
    frames: list[tts.SynthesizedAudio] = []
    async for frame in stream:
        frames.append(frame)
    return frames


async def test_stream_with_timestamps() -> None:
    """https://github.com/livekit/agents/pull/7543"""
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
    words = {t.text: t for t in transcripts}
    assert set(words) == {"Hello", "world"}
    assert words["Hello"].start_time == pytest.approx(0.0)
    assert words["Hello"].end_time == pytest.approx(0.32)
    assert words["world"].start_time == pytest.approx(0.34)

    # the body is read past speech.done, so its connection can go back to the pool
    assert client.audio.streams_read_to_end == 1
    # speech.done's billed characters ride on the segment's final frame
    assert [f.characters_count for f in frames if f.characters_count is not None] == [11]
    assert frames[-1].is_final and frames[-1].characters_count == 11


async def test_stream_predicates_speech_error() -> None:
    """https://github.com/livekit/agents/pull/6926"""
    client = _FakeClient()
    client.audio = _ErrorAudio()
    stream = _make_tts("simba-3.2", client).stream(conn_options=APIConnectOptions(retry_interval=0))
    stream.push_text("Hello world.")
    stream.end_input()

    with pytest.raises(APIStatusError, match="boom"):
        await _collect(stream)


async def test_synthesize_streams_with_timestamps() -> None:
    """https://github.com/livekit/agents/pull/7543"""
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
    assert [t.text for t in transcripts] == ["Hello", "world"]
    assert transcripts[1].start_time == pytest.approx(0.34)
