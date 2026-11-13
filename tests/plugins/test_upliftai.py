"""Uplift AI TTS: provider errors fail synthesis, against a local Socket.IO peer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import socketio  # type: ignore[import-untyped]  # python-socketio ships no type information
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit.agents import APIConnectOptions, APIError
from livekit.plugins.upliftai.tts import TTS, WEBSOCKET_NAMESPACE
from tests.fakes import until


@pytest.fixture
async def uplift() -> AsyncIterator[tuple[TTS, list[dict[str, object]]]]:
    """A TTS on a local Uplift peer that answers each synthesis with the next queued message."""
    replies: list[dict[str, object]] = []
    server = socketio.AsyncServer(async_mode="aiohttp", ping_interval=0.01)
    app = web.Application()
    server.attach(app)

    # python-socketio's decorators are untyped
    @server.on("connect", namespace=WEBSOCKET_NAMESPACE)  # type: ignore[untyped-decorator]
    async def connected(sid: str, environ: dict[str, object], auth: object) -> None:
        await server.emit("message", {"type": "ready"}, to=sid, namespace=WEBSOCKET_NAMESPACE)

    @server.on("synthesize", namespace=WEBSOCKET_NAMESPACE)  # type: ignore[untyped-decorator]
    async def synthesize(sid: str, data: dict[str, str]) -> None:
        reply = {**replies.pop(0), "requestId": data["requestId"]}
        await server.emit("message", reply, to=sid, namespace=WEBSOCKET_NAMESPACE)

    async with TestServer(app) as peer:
        provider = TTS(api_key="key", base_url=str(peer.make_url("")), output_format="PCM_22050_16")
        yield provider, replies
        await provider.aclose()
        await server.shutdown()
        # Engine.IO cancels its scheduled ping after shutdown() returns
        await until(
            lambda: not [t for t in asyncio.all_tasks() if "_send_ping" in repr(t.get_coro())],
            "Engine.IO's ping task ended",
        )


async def _synthesize(provider: TTS) -> None:
    async with provider.synthesize("hello", conn_options=APIConnectOptions(max_retry=0)) as stream:
        async for _ in stream:
            pytest.fail("a failed synthesis produced audio")


async def test_provider_rejection_fails_synthesis(
    uplift: tuple[TTS, list[dict[str, object]]],
) -> None:
    """A provider error fails synthesis for good, carrying the provider's message."""
    provider, replies = uplift
    replies.append({"type": "error", "message": "voice unavailable"})

    with pytest.raises(APIError) as error:
        await _synthesize(provider)

    assert error.value.retryable is False
    assert isinstance(error.value.body, dict)
    assert error.value.body["message"] == "voice unavailable"


async def test_malformed_socket_message_fails_pending_synthesis_without_waiting_for_timeout(
    uplift: tuple[TTS, list[dict[str, object]]],
) -> None:
    """A malformed audio message fails the pending synthesis at once, not at its timeout."""
    provider, replies = uplift
    replies.append({"type": "audio", "audio": []})

    with pytest.raises(APIError) as failure:
        async with asyncio.timeout(1):
            await _synthesize(provider)

    assert failure.value.retryable is False
    assert isinstance(failure.value.__cause__, ValidationError)
