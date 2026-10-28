"""TTS upgrades respect the request's handshake read limit."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APITimeoutError

pytestmark = pytest.mark.unit


@pytest.fixture
async def silent_upgrade() -> AsyncIterator[str]:
    handlers: list[asyncio.Task[None]] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        handlers.append(asyncio.create_task(handle(reader, writer)))

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    try:
        yield f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        await server.wait_closed()
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.parametrize("provider", ["elevenlabs", "soniox", "baseten", "gradium", "minimax"])
async def test_silent_tts_upgrade_reports_read_timeout(provider: str, silent_upgrade: str) -> None:
    async with aiohttp.ClientSession() as session:
        if provider == "elevenlabs":
            from livekit.plugins.elevenlabs import TTS

            tts = TTS(
                api_key="test", base_url=silent_upgrade, encoding="pcm_24000", http_session=session
            )
        elif provider == "soniox":
            from livekit.plugins.soniox import TTS

            tts = TTS(api_key="test", websocket_url=silent_upgrade, http_session=session)
        elif provider == "gradium":
            from livekit.plugins.gradium import TTS

            tts = TTS(api_key="test", model_endpoint=silent_upgrade, http_session=session)
        elif provider == "minimax":
            from livekit.plugins.minimax import TTS

            tts = TTS(
                api_key="test", base_url=silent_upgrade, audio_format="pcm", http_session=session
            )
        else:
            from livekit.plugins.baseten import TTS

            tts = TTS(
                model="qwen3-tts",
                api_key="test",
                model_endpoint=silent_upgrade,
                voice="test",
                http_session=session,
            )
        try:
            options = APIConnectOptions(
                max_retry=0,
                pool_timeout=None,
                connect_timeout=None,
                write_timeout=None,
                read_timeout=0.02,
            )
            async with tts.stream(conn_options=options) as stream:
                stream.push_text("Hello there.")
                stream.end_input()
                async with asyncio.timeout(0.5):
                    with pytest.raises(APITimeoutError) as error:
                        async for _ in stream:
                            pass
                assert error.value.phase == "read"
        finally:
            await tts.aclose()


async def test_rtzr_silent_upgrade_reports_read_timeout(silent_upgrade: str) -> None:
    import time

    from livekit.plugins.rtzr.rtzrapi import RTZROpenAPIClient

    options = APIConnectOptions(
        max_retry=0,
        pool_timeout=None,
        connect_timeout=None,
        write_timeout=None,
        read_timeout=0.02,
    )
    async with aiohttp.ClientSession() as session:
        client = RTZROpenAPIClient("test", "test", http_session=session)
        client._token = {"access_token": "cached", "expire_at": time.time() + 10000}
        client._ws_base = silent_upgrade
        async with asyncio.timeout(0.5):
            with pytest.raises(APITimeoutError) as error:
                await client.connect_websocket({}, conn_options=options)
        assert error.value.phase == "read"
