"""The host protocols carry each real VoiceSession request onto its original SpeechHandle."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import pytest
from aiohttp import web

from livekit.agents import Agent, VoiceSession
from livekit.plugins import phonic, ultravox
from tests.fakes import FakeAudioOutput

if TYPE_CHECKING:
    from tests.plugins.conftest import Connection, Frame, WsPeer


@pytest.mark.parametrize("provider", ["phonic", "ultravox"])
async def test_two_requested_responses_commit_to_their_own_returned_speech_handles(
    provider: str,
    ws_peer: WsPeer,
) -> None:
    replied = 0
    pcm = b"\x00\x01" * 240

    def answer(_connection: Connection, data: str | bytes) -> list[Frame]:
        nonlocal replied
        if isinstance(data, bytes):
            return []
        event = json.loads(data)
        kind = event["type"]
        if provider == "phonic" and kind == "config":
            return [{"type": "ready_to_start_conversation"}]
        requested = (
            kind == "generate_reply"
            if provider == "phonic"
            else (kind == "user_text_message" and not event.get("deferResponse", True))
        )
        if not requested:
            return []
        replied += 1
        text = f"reply number {replied}"
        if provider == "phonic":
            return [
                {"type": "assistant_started_speaking"},
                {"type": "audio_chunk", "audio": base64.b64encode(pcm).decode(), "text": text},
                {"type": "assistant_finished_speaking"},
            ]
        return [
            {"type": "state", "state": "thinking"},
            {
                "type": "transcript",
                "role": "agent",
                "text": text,
                "final": False,
                "ordinal": replied,
            },
            {"type": "state", "state": "speaking"},
            pcm,
            {
                "type": "transcript",
                "role": "agent",
                "text": text,
                "final": True,
                "ordinal": replied,
            },
        ]

    ws_peer.reply = answer
    if provider == "phonic":
        model = phonic.realtime.RealtimeModel(api_key="key", base_url=ws_peer.ws_url)
    else:

        async def create_call(_request: web.Request) -> web.Response:
            return web.json_response({"joinUrl": ws_peer.ws_url + "/join"})

        ws_peer.http = create_call
        model = ultravox.realtime.RealtimeModel(api_key="key", base_url=ws_peer.url + "/api")
    try:
        async with VoiceSession(llm=model, vad=None) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="Be brief."))
            first = session.generate_reply(instructions="first request")
            second = session.generate_reply(instructions="second request")
            await first
            await second
            assert first.done() and second.done()
            assert first.exception() is second.exception() is None
            assert first is not second
            assert len(first.chat_items) == len(second.chat_items) == 1
            assert first.chat_items[0].text_content == "reply number 1"
            assert second.chat_items[0].text_content == "reply number 2"
            assert first.chat_items[0].id != second.chat_items[0].id
    finally:
        await model.aclose()


@pytest.mark.parametrize("provider", ["phonic", "ultravox"])
@pytest.mark.parametrize("started", [False, True])
async def test_a_terminal_provider_error_settles_its_request_or_open_generation(
    provider: str,
    started: bool,
    ws_peer: WsPeer,
) -> None:
    import asyncio

    from livekit.agents import APIConnectOptions, llm
    from tests.fakes import until
    from tests.plugins.conftest import LivePeer

    ws_peer.reply = lambda _c, data: (
        [{"type": "ready_to_start_conversation"}]
        if isinstance(data, str) and json.loads(data)["type"] == "config"
        else []
    )
    if provider == "phonic":
        model = phonic.realtime.RealtimeModel(
            api_key="key", base_url=ws_peer.ws_url, conn_options=APIConnectOptions(max_retry=0)
        )
    else:

        async def create_call(_request: web.Request) -> web.Response:
            return web.json_response({"joinUrl": ws_peer.ws_url + "/join"})

        ws_peer.http = create_call
        model = ultravox.realtime.RealtimeModel(
            api_key="key",
            base_url=ws_peer.url + "/api",
            system_prompt="Be brief.",
            conn_options=APIConnectOptions(max_retry=0),
        )
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    drained: asyncio.Task[tuple[str, int, list[str]]] | None = None
    try:
        await session.update_instructions("Be brief.")
        await session.update_tools([])
        connection = await ws_peer.connected()
        pending = session.generate_reply()
        if started:
            pcm = b"\x00\x01" * 240
            if provider == "phonic":
                await ws_peer.send(
                    {"type": "assistant_started_speaking"},
                    {
                        "type": "audio_chunk",
                        "audio": base64.b64encode(pcm).decode(),
                        "text": "partial",
                    },
                )
            else:
                await ws_peer.send(
                    {"type": "state", "state": "thinking"},
                    {
                        "type": "transcript",
                        "role": "agent",
                        "text": "partial",
                        "final": False,
                        "ordinal": 1,
                    },
                    pcm,
                )
            generation = await pending
            drained = asyncio.create_task(LivePeer.drain(generation))
        await connection.socket.close()
        await until(lambda: errors, "the provider failure is reported")
        if drained is None:
            assert pending.done(), "the failed request must not wait for its timer"
            with pytest.raises(llm.RealtimeError):
                pending.result()
        else:
            await until(drained.done, "the failed generation releases its streams")
            assert await drained == ("partial", 1, [])
        assert not errors[-1].recoverable
    finally:
        if drained is not None:
            drained.cancel()
            await asyncio.gather(drained, return_exceptions=True)
        await session.aclose()
        await model.aclose()
