"""Sarvam STT (batch and saaras streaming) against a local peer speaking Sarvam's wire."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, http, stt
from livekit.plugins.sarvam import stt as sarvam_stt
from tests.fakes import until

NO_RETRY = APIConnectOptions(max_retry=0)
Peer = Callable[[web.WebSocketResponse], Awaitable[None]]


@asynccontextmanager
async def _streaming_stt(
    monkeypatch: pytest.MonkeyPatch, peer: Peer
) -> AsyncIterator[sarvam_stt.STT]:
    """A saaras:v3 STT whose streams connect to `peer` (the plugin takes no streaming URL)."""

    async def handle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await peer(ws)
        return ws

    app = web.Application()
    app.router.add_get("/ws", handle)
    async with TestServer(app) as server:
        url = str(server.make_url("/ws")).replace("http", "ws", 1)
        monkeypatch.setattr(sarvam_stt, "SARVAM_STT_STREAMING_URL", url)
        model = sarvam_stt.STT(api_key="sk_test", model="saaras:v3")
        try:
            yield model
        finally:
            await model.aclose()


def _frame(ms: int) -> rtc.AudioFrame:
    samples = 16 * ms
    return rtc.AudioFrame(b"\x01\x00" * samples, 16000, 1, samples)


def _kind(message: dict[str, object]) -> str:
    return str(message.get("type", "audio"))


async def test_final_carries_provider_confidence_and_speech_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final transcript keeps Sarvam's confidence and speech bounds, ahead of EOS.

    https://github.com/livekit/agents/issues/5829 https://github.com/livekit/agents/pull/6052
    """

    async def peer(ws: web.WebSocketResponse) -> None:
        await ws.send_json({"type": "events", "data": {"signal_type": "START_SPEECH"}})
        await ws.send_json(
            {
                "type": "data",
                "data": {
                    "transcript": "namaste",
                    "language_code": "hi-IN",
                    "language_probability": 0.42,
                    "speech_start": 0.5,
                    "speech_end": 1.25,
                },
            }
        )
        await ws.send_json({"type": "events", "data": {"signal_type": "END_SPEECH"}})
        async for _ in ws:
            pass

    async with _streaming_stt(monkeypatch, peer) as model:
        async with model.stream(conn_options=NO_RETRY) as stream:
            events = []
            async for event in stream:
                if event.type != stt.SpeechEventType.RECOGNITION_USAGE:
                    events.append(event)
                if event.type == stt.SpeechEventType.END_OF_SPEECH:
                    break
            offset = stream.start_time_offset

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = events[1].alternatives[0]
    assert final.text == "namaste"
    assert final.confidence == pytest.approx(0.42)
    assert (final.start_time, final.end_time) == pytest.approx((0.5 + offset, 1.25 + offset))


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/7990 reproduces: flush ends the stream",
)
async def test_flush_drains_audio_and_keeps_the_stream_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flush sends buffered audio and later audio still flows; only end_input ends it.

    https://github.com/livekit/agents/pull/7990
    """
    received: asyncio.Queue[str] = asyncio.Queue()

    async def peer(ws: web.WebSocketResponse) -> None:
        async for incoming in ws:
            received.put_nowait(_kind(incoming.json()))
            if _kind(incoming.json()) == "end_of_stream":
                await ws.close()

    async with _streaming_stt(monkeypatch, peer) as model:
        async with model.stream(conn_options=NO_RETRY) as stream:
            stream.push_frame(_frame(30))  # below one 50 ms chunk: only a flush sends it
            stream.flush()
            assert await asyncio.wait_for(received.get(), 2) == "audio"
            stream.push_frame(_frame(50))
            assert await asyncio.wait_for(received.get(), 2) == "audio"
            stream.end_input()
            assert await asyncio.wait_for(received.get(), 2) == "end_of_stream"


async def test_saaras_v3_prompt_sends_no_config_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """saaras:v3's endpoint accepts no config frame, so a prompt never sends one.

    https://github.com/livekit/agents/issues/6606
    """
    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def peer(ws: web.WebSocketResponse) -> None:
        async for incoming in ws:
            received.put_nowait(incoming.json())

    async with _streaming_stt(monkeypatch, peer) as model:
        async with model.stream(prompt="guests, two, three", conn_options=NO_RETRY) as stream:
            stream.push_frame(_frame(50))
            first = await received.get()

    assert _kind(first) == "audio"


async def test_an_error_frame_raises_with_its_status_and_the_raw_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider's error frame reaches the caller whole, its code as the status."""
    frame = {
        "type": "error",
        "data": {"message": "invalid model", "code": "400", "details": "model not found"},
    }

    async def peer(ws: web.WebSocketResponse) -> None:
        await ws.send_json(frame)
        async for _ in ws:
            pass

    async with _streaming_stt(monkeypatch, peer) as model:
        with pytest.raises(APIStatusError) as caught:
            async with model.stream(conn_options=NO_RETRY) as stream:
                async for _ in stream:
                    pass

    assert caught.value.status_code == 400
    assert caught.value.body == frame


async def test_a_rejecting_close_raises_with_the_close_code_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4000 close (e.g. an invalid model) is an error, not a quiet end of stream."""

    async def peer(ws: web.WebSocketResponse) -> None:
        await ws.close(code=4000, message=b"Invalid model 'saarass:v3'.")

    async with _streaming_stt(monkeypatch, peer) as model:
        with pytest.raises(APIStatusError) as caught:
            async with model.stream(conn_options=NO_RETRY) as stream:
                async for _ in stream:
                    pass

    assert caught.value.status_code == 4000
    assert caught.value.body == {
        "msg_type": "CLOSE",
        "close_code": 4000,
        "close_reason": "Invalid model 'saarass:v3'.",
    }


async def test_closing_the_stt_closes_every_open_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://github.com/livekit/agents/pull/7058"""
    heard: asyncio.Queue[None] = asyncio.Queue()
    closed: asyncio.Queue[None] = asyncio.Queue()

    async def peer(ws: web.WebSocketResponse) -> None:
        async for _ in ws:
            heard.put_nowait(None)
        closed.put_nowait(None)

    async with _streaming_stt(monkeypatch, peer) as model:
        for language in ("en-IN", "hi-IN"):
            model.stream(language=language, conn_options=NO_RETRY).push_frame(_frame(50))
            await heard.get()  # connected and streaming

        await model.aclose()

        await closed.get()
        await closed.get()


@pytest.mark.parametrize("as_list", [False, True])
async def test_batch_transcript_without_timestamps_ends_at_the_audio_duration(
    as_list: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def recognize(request: web.Request) -> web.Response:
        await request.read()
        return web.json_response({"transcript": "hello", "language_code": "en-IN"})

    app = web.Application()
    app.router.add_post("/recognize", recognize)
    async with TestServer(app) as server, http.open():
        # recognize() reads the module URL; the base_url argument does not reach it
        monkeypatch.setattr(sarvam_stt, "SARVAM_STT_BASE_URL", str(server.make_url("/recognize")))
        model = sarvam_stt.STT(api_key="test")
        frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=1600)
        result = await model.recognize([frame] if as_list else frame)
        await model.aclose()

    assert result.alternatives[0].end_time == pytest.approx(0.1)


@asynccontextmanager
async def _sarvam_and_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[str, list[str]]]:
    """Serves Sarvam's default endpoints under /default and a custom one under /custom.

    Yields the custom base_url and the path of every request received.
    """
    paths: list[str] = []

    async def recognize(request: web.Request) -> web.Response:
        paths.append(request.path)
        await request.read()
        return web.json_response({"transcript": "hello", "language_code": "en-IN"})

    async def listen(request: web.Request) -> web.WebSocketResponse:
        paths.append(request.path)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    for root in ("/default", "/custom"):
        app.router.add_post(f"{root}/speech-to-text", recognize)
        app.router.add_get(f"{root}/speech-to-text/ws", listen)
    async with TestServer(app) as server, http.open():
        default = str(server.make_url("/default/speech-to-text"))
        monkeypatch.setattr(sarvam_stt, "SARVAM_STT_BASE_URL", default)
        monkeypatch.setattr(
            sarvam_stt, "SARVAM_STT_STREAMING_URL", default.replace("http", "ws", 1) + "/ws"
        )
        yield str(server.make_url("/custom/speech-to-text")), paths


@pytest.mark.xfail(strict=True, reason="recognize() posts to Sarvam's URL, never base_url")
async def test_recognize_posts_to_the_given_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch recognition goes to the ``base_url`` the STT was built with."""
    async with _sarvam_and_custom_endpoint(monkeypatch) as (base_url, paths):
        model = sarvam_stt.STT(api_key="sk_test", model="saarika:v2.5", base_url=base_url)
        await model.recognize(_frame(100), conn_options=NO_RETRY)
        await model.aclose()

    assert paths == ["/custom/speech-to-text"]


@pytest.mark.xfail(strict=True, reason="stream() connects to Sarvam's URL, never base_url")
async def test_stream_connects_to_the_given_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A streaming recognition connects under the ``base_url`` the STT was built with."""
    async with _sarvam_and_custom_endpoint(monkeypatch) as (base_url, paths):
        model = sarvam_stt.STT(api_key="sk_test", model="saarika:v2.5", base_url=base_url)
        async with model.stream(conn_options=NO_RETRY) as stream:
            stream.push_frame(_frame(100))
            await until(lambda: paths, "the stream connects")
        await model.aclose()

    assert paths == ["/custom/speech-to-text/ws"]
