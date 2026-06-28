"""Unit tests for Blaze STT plugin."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.stt import SpeechEventType
from livekit.plugins.blaze._config import BlazeConfig
from livekit.plugins.blaze.stt import STT

pytestmark = pytest.mark.unit


def _pcm_frame(
    *,
    samples: int = 1600,
    sample_rate: int = 16000,
    byte_val: int = 0x11,
) -> rtc.AudioFrame:
    data = bytes([byte_val, byte_val]) * samples
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _QueueTransport(httpx.AsyncBaseTransport):
    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._responder(request)


def _make_stt(
    responder: Callable[[httpx.Request], httpx.Response],
    *,
    normalization_rules: dict[str, str] | None = None,
) -> STT:
    config = BlazeConfig(api_url="https://api.example.com", api_token="test-token")
    stt = STT(config=config, normalization_rules=normalization_rules)
    stt._client = httpx.AsyncClient(transport=_QueueTransport(responder))
    return stt


@pytest.mark.asyncio
async def test_stt_provider_and_sample_rate() -> None:
    stt = _make_stt(lambda _req: httpx.Response(200, json={"transcription": "ok"}))
    assert stt.provider == "Blaze"
    assert stt.sample_rate == 16000
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_empty_buffer_returns_empty_event() -> None:
    stt = _make_stt(lambda _req: httpx.Response(200, json={"transcription": "unused"}))

    event = await stt._recognize_impl([], conn_options=APIConnectOptions(max_retry=0))

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives == []
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_returns_transcription_with_normalization() -> None:
    stt = _make_stt(
        lambda _req: httpx.Response(200, json={"transcription": "API", "confidence": 0.88}),
        normalization_rules={"API": "A P I"},
    )

    event = await stt._recognize_impl(_pcm_frame(), conn_options=APIConnectOptions(max_retry=0))

    assert event.alternatives[0].text == "A P I"
    assert event.alternatives[0].confidence == 0.88
    assert stt._pending_pcm == b""
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_buffers_empty_transcription_for_next_call() -> None:
    request_count = 0

    def responder(_req: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(200, json={"transcription": "", "confidence": 0.0})
        return httpx.Response(200, json={"transcription": "xin chào", "confidence": 0.9})

    stt = _make_stt(responder)
    frame_a = _pcm_frame(samples=800, byte_val=0x01)
    frame_b = _pcm_frame(samples=800, byte_val=0x02)

    first = await stt._recognize_impl(frame_a, conn_options=APIConnectOptions(max_retry=0))
    assert first.alternatives[0].text == ""
    assert stt._pending_pcm != b""
    assert stt._pending_empty_count == 1

    second = await stt._recognize_impl(frame_b, conn_options=APIConnectOptions(max_retry=0))
    assert second.alternatives[0].text == "xin chào"
    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    assert request_count == 2
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_discards_pending_after_max_empty_segments() -> None:
    def responder(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"transcription": "", "confidence": 0.0})

    stt = _make_stt(responder)
    stt._max_pending_segments = 2

    for _ in range(3):
        await stt._recognize_impl(
            _pcm_frame(samples=400), conn_options=APIConnectOptions(max_retry=0)
        )

    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_discards_pending_when_duration_limit_exceeded() -> None:
    def responder(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"transcription": "", "confidence": 0.0})

    stt = _make_stt(responder)
    stt._max_pending_duration = 0.05

    # ~0.1s of audio at 16kHz mono PCM16
    large_frame = _pcm_frame(samples=1600)

    await stt._recognize_impl(large_frame, conn_options=APIConnectOptions(max_retry=0))

    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_clears_stale_pending_buffer_after_idle_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stt = _make_stt(
        lambda _req: httpx.Response(200, json={"transcription": "ok", "confidence": 1.0})
    )
    stt._pending_pcm = b"\x01\x00" * 100
    stt._pending_empty_count = 1
    stt._last_recognize_time = time.monotonic() - 20.0
    stt._pending_idle_timeout = 10.0

    await stt._recognize_impl(_pcm_frame(samples=200), conn_options=APIConnectOptions(max_retry=0))

    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_raises_api_status_error_on_http_failure() -> None:
    from livekit.agents import APIStatusError

    stt = _make_stt(lambda _req: httpx.Response(500, text="server error"))

    with pytest.raises(APIStatusError, match="STT service error 500"):
        await stt._recognize_impl(_pcm_frame(), conn_options=APIConnectOptions(max_retry=0))

    await stt.aclose()


def test_stt_with_streaming_requires_vad_instance() -> None:
    stt = STT(config=BlazeConfig(api_url="https://api.example.com"))

    with pytest.raises(TypeError, match="Expected a VAD instance"):
        stt.with_streaming(object())  # type: ignore[arg-type]
