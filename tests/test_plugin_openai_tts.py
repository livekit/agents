from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import openai
import pytest

from livekit.plugins.openai import TTS

pytestmark = pytest.mark.unit

PCM = b"\x00\x01" * 4800  # 200ms of 16-bit mono at 24kHz


def _tts(handler, *, model: str, response_format: str = "pcm") -> TTS:
    client = openai.AsyncClient(
        api_key="test",
        base_url="https://compatible.example.com/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return TTS(model=model, client=client, response_format=response_format)


async def _synthesize(tts: TTS) -> bytes:
    audio = b""
    async with tts.synthesize("hello") as stream:
        async for ev in stream:
            audio += ev.frame.data.tobytes()
    return audio


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
