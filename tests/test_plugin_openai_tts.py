from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

import httpx2 as httpx
import openai
import pytest

from livekit.plugins.openai import TTS

pytestmark = pytest.mark.unit

PCM = b"\x00\x01" * 4800  # 200ms of 16-bit mono at 24kHz


def _tts(handler, *, model: str, response_format: str = "pcm", sample_rate: int = 24000) -> TTS:
    client = openai.AsyncClient(
        api_key="test",
        base_url="https://compatible.example.com/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return TTS(model=model, client=client, response_format=response_format, sample_rate=sample_rate)


async def _synthesize(tts: TTS) -> bytes:
    audio = b""
    async with tts.synthesize("hello") as stream:
        async for ev in stream:
            audio += ev.frame.data.tobytes()
    return audio


@pytest.mark.parametrize(
    "sample_rate", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.5, float("nan"), float("inf")]
)
def test_rejects_invalid_sample_rate_before_synthesis(sample_rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate must be an integer of at least 10 Hz"):
        TTS(api_key="test", sample_rate=sample_rate)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sample_rate", "expected_rate"),
    [(None, 24000), (10, 10), (44100, 44100)],
)
async def test_preserves_pcm_and_frame_metadata_for_sample_rate(
    sample_rate: int | None, expected_rate: int
) -> None:
    samples = bytes(i % 256 for i in range((expected_rate // 5 + 1) * 2))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=samples, headers={"content-type": "audio/pcm"})

    kwargs = {} if sample_rate is None else {"sample_rate": sample_rate}
    tts = _tts(handler, model="tts-1", **kwargs)
    events = []
    async with tts.synthesize("hello") as stream:
        async for event in stream:
            events.append(event)

    assert tts.sample_rate == expected_rate
    frames = [event.frame for event in events]
    assert frames
    assert all(frame.sample_rate == expected_rate for frame in frames)
    assert all(frame.num_channels == 1 for frame in frames)
    audio = b"".join(frame.data.tobytes() for frame in frames)
    assert audio[: len(samples)] == samples
    assert not any(audio[len(samples) :])
    assert not any(event.is_final for event in events[:-1])
    assert events[-1].is_final


@pytest.mark.parametrize("model", ["hexgrad/Kokoro-82M", "gpt-4o-mini-tts"])
async def test_audio_body_is_decoded_whatever_the_model(model: str) -> None:
    """A compatible server ignores stream_format and answers with plain audio bytes."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PCM, headers={"content-type": "audio/pcm"})

    audio = await _synthesize(_tts(handler, model=model))

    assert audio[: len(PCM)] == PCM


async def test_declared_content_type_wins_over_requested_format() -> None:
    """A server may ignore response_format; decode what it says it sent, not what we asked for."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(PCM)
    wav = buf.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=wav, headers={"content-type": "audio/wav"})

    # mp3 is requested, but the server answers with wav and says so
    audio = await _synthesize(_tts(handler, model="kokoro", response_format="mp3"))

    assert audio


async def test_sse_body_is_parsed() -> None:
    """OpenAI answers stream_format="sse" with an event stream, which is still parsed."""

    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            json.dumps({"type": "speech.audio.delta", "delta": base64.b64encode(PCM).decode()}),
            json.dumps({"type": "speech.audio.done", "usage": {"input_tokens": 1}}),
        ]
        body = "".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    audio = await _synthesize(_tts(handler, model="gpt-4o-mini-tts"))

    assert audio[: len(PCM)] == PCM


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("tts-1", "audio"),
        ("tts-1-hd", "audio"),
        ("gpt-4o-mini-tts", "sse"),
    ],
)
async def test_stream_format_requested_per_model(model: str, expected: str) -> None:
    """`sse` is not supported for tts-1/tts-1-hd, so those must keep requesting `audio`."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=PCM, headers={"content-type": "audio/pcm"})

    await _synthesize(_tts(handler, model=model))

    assert requests[0]["stream_format"] == expected


async def test_a_second_prewarm_does_not_replace_one_still_in_flight() -> None:
    """aclose() cancels only the task it holds, so a replaced prewarm outlives the close."""
    release = asyncio.Event()
    in_flight = asyncio.Semaphore(0)

    async def handler(request: httpx.Request) -> httpx.Response:
        in_flight.release()
        await release.wait()
        return httpx.Response(200, content=b"", headers={"content-type": "text/plain"})

    tts = _tts(handler, model="gpt-4o-mini-tts")
    tts.prewarm()
    first = tts._prewarm_task
    await in_flight.acquire()  # the first request is genuinely open
    tts.prewarm()

    await tts.aclose()
    try:
        assert first is not None and first.done(), "a prewarm task outlived aclose()"
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
