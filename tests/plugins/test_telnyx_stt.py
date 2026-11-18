"""Telnyx STT shutdown against a local endpoint speaking its engine-specific protocol."""

import json

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, http, stt
from livekit.plugins.telnyx import STT


async def test_deepgram_engine_flushes_trailing_audio_before_server_close() -> None:
    controls: list[str] = []

    async def recognize(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for raw in socket:
            if raw.type != web.WSMsgType.TEXT:
                continue
            control = json.loads(raw.data)["type"]
            controls.append(control)
            if control == "CloseStream":
                await socket.send_json({"transcript": "trailing words", "is_final": True})
                await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/stt", recognize)
    async with TestServer(app) as server, http.open():
        model = STT(
            api_key="key", transcription_engine="deepgram", base_url=str(server.make_url("/stt"))
        )
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_frame(rtc.AudioFrame(bytes(3200), 16000, 1, 1600))
            stream.end_input()
            finals = [
                event.alternatives[0].text
                async for event in stream
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            ]
        await model.aclose()
    assert controls == ["CloseStream"]
    assert finals == ["trailing words"]
