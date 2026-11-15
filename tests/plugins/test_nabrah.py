"""Nabrah STT plugin: provider error redaction and the end of a turn, against a local peer."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIError, _core, http, stt
from livekit.plugins.nabrah import STT
from tests.fakes import until
from tests.plugins.conftest import WsPeer

PREFLIGHT = stt.SpeechEventType.PREFLIGHT_TRANSCRIPT
FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT

TRANSCRIPT = "private customer transcript"


async def _listen(request: web.Request) -> web.WebSocketResponse:
    socket = web.WebSocketResponse()
    await socket.prepare(request)
    await socket.receive()  # the configuration frame
    await socket.send_json({"type": "ready"})
    await socket.send_json({"type": "error", "message": TRANSCRIPT})
    async for _ in socket:
        pass
    return socket


async def test_provider_error_does_not_expose_provider_message() -> None:
    """A provider error fails the stream without its message. https://github.com/livekit/agents/pull/7337"""
    app = web.Application()
    app.router.add_get("/listen", _listen)
    async with TestServer(app) as server, http.open():
        provider = STT(api_key="key", base_url=str(server.make_url("/listen").with_scheme("ws")))

        with pytest.raises(APIError) as caught:
            async with provider.stream() as stream:
                async for _ in stream:
                    pass

    assert TRANSCRIPT not in str(caught.value)


def _audio(seconds: float) -> rtc.AudioFrame:
    samples = round(16000 * seconds)
    return rtc.AudioFrame(b"\0\0" * samples, 16000, 1, samples)


async def test_an_end_of_turn_token_ends_the_turn_once_audio_or_silence_confirms_it(
    ws_peer: WsPeer,
) -> None:
    """An <eot> ends the turn after the confirm delay's worth of audio with no new words, or the
    same time with no audio at all; until then it is only a preflight."""
    ws_peer.hello = [{"type": "ready"}]
    provider = STT(api_key="key", base_url=ws_peer.ws_url)
    events: list[stt.SpeechEvent] = []

    def said(kind: stt.SpeechEventType) -> list[str]:
        return [e.alternatives[0].text for e in events if e.type == kind]

    async with provider.stream() as stream:
        reader = asyncio.create_task(_collect(stream, events))
        stream.push_frame(_audio(0.1))
        await ws_peer.connected()
        await ws_peer.send({"type": "transcript", "text": "hello<eot>"})
        await until(lambda: said(PREFLIGHT) == ["hello"], "the preflight")
        stream.push_frame(_audio(0.2))
        stream.push_frame(_audio(0.3))
        await until(lambda: said(FINAL) == ["hello"], "the audio confirms the turn's end")

        await ws_peer.send({"type": "transcript", "text": "hello again<eot>"})
        await until(lambda: said(PREFLIGHT) == ["hello", "again"], "the second preflight")
        stream.push_frame(_audio(0.01))
        await until(lambda: len(ws_peer.connections[0].received) == 5, "the last audio")
        _core.skip_host_time(0.4)
        await until(lambda: said(FINAL) == ["hello", "again"], "the silence confirms it")
        reader.cancel()

    kinds = [e.type for e in events if e.type in (stt.SpeechEventType.END_OF_SPEECH, FINAL)]
    assert kinds == [FINAL, stt.SpeechEventType.END_OF_SPEECH] * 2


async def _collect(stream: stt.RecognizeStream, into: list[stt.SpeechEvent]) -> None:
    async for event in stream:
        into.append(event)
