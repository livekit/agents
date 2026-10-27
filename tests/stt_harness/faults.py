"""Optional deterministic transport fixtures, owned by a discovered provider."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class FaultCase:
    name: str
    module: str
    stream_class: str = "SpeechStream"
    constructor: str = "STT"
    options: dict[str, Any] = field(default_factory=dict)
    greetings: tuple[dict[str, Any], ...] = ()
    completion: tuple[dict[str, Any] | bytes, ...] = ()
    constructor_module: str | None = None
    reply: Callable | None = None
    stop: dict[str, Any] | str | bytes | None = None
    completion_closes_socket: bool = True
    api_key: bool = True
    http_session: bool = True

    def create(self, session):
        module = import_module(self.constructor_module or self.module)
        kwargs = dict(self.options)
        if self.api_key:
            kwargs["api_key"] = "test-key"
        if self.http_session:
            kwargs["http_session"] = session
        return getattr(module, self.constructor)(**kwargs)

    @property
    def stream_type(self):
        return getattr(import_module(self.module), self.stream_class)


def plugin(name):
    return f"livekit.plugins.{name}"


def _gladia_reply(socket, data):
    if json.loads(data).get("type") == "stop_recording":
        for event in socket.transport.case.completion:
            socket.emit(event)


def _microsoft_reply(socket, data):
    if json.loads(data).get("type") == "input_audio_buffer.commit":
        item_id = f"item-{len(socket.writes)}"
        socket.emit({"type": "input_audio_buffer.committed", "item_id": item_id})
        socket.emit(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": "words",
            }
        )


def _rtzr_reply(socket, data):
    if data == "EOS":
        socket.server_close()


def make_provider(case, monkeypatch):
    import aiohttp

    from livekit.agents import utils
    from tests.fake_stt_transport import Transport

    transport = Transport(case)

    async def unexpected_request(*args, **kwargs):
        raise AssertionError("STT contract tests must not use the network")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", unexpected_request)
    monkeypatch.setenv("RTZR_CLIENT_ID", "test")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "test")
    monkeypatch.setattr(aiohttp, "ClientSession", type(transport.session()))
    monkeypatch.setattr(utils.http_context, "http_session", transport.session)
    instance = case.create(transport.session())
    return case, instance, transport


async def assert_control_write_failure(case, monkeypatch, message, action):
    """Check provider-specific stop/flush packets with the shared transport."""
    import asyncio
    import json

    import aiohttp
    import pytest

    from livekit import rtc
    from livekit.agents import APIConnectionError, APIConnectOptions
    from tests.fake_stt_transport import Socket

    from .drivers import collect_events

    cause = aiohttp.ClientConnectionError("connection reset")
    original_send = Socket.send_str

    async def send(self, data):
        event = json.loads(data)
        if message in event or event.get("type") == message:
            raise cause
        await original_send(self, data)

    monkeypatch.setattr(Socket, "send_str", send)
    _, provider, _ = make_provider(case, monkeypatch)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    collector = asyncio.create_task(collect_events(stream, []))
    try:
        if action == "audio":
            stream.push_frame(rtc.AudioFrame.create(16000, 1, 160))
        elif action == "flush":
            stream.flush()
        else:
            stream.end_input()
        with pytest.raises(APIConnectionError) as info:
            await asyncio.wait_for(collector, 1)
        assert info.value.retryable
        assert info.value.__cause__ is cause
    finally:
        await stream.aclose()
        await asyncio.gather(collector, return_exceptions=True)
        await provider.aclose()
