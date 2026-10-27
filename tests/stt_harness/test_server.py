"""The server must detect bad clients and preserve real WebSocket behavior."""

import asyncio

import aiohttp
import pytest

from .protocol import Protocol, Provider
from .server import Segment, STTServer, json_message

pytestmark = pytest.mark.unit


class Example(Protocol):
    async def open(self, request):
        self.configured = True

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            assert json_message(message) == {"type": "stop"}
            peer.end_input()
            await peer.finalize()
            peer.close()

    async def transcript(self, result):
        self.connection.send({"text": result.text, "final": result.final}, final=result.final)


CASE = Provider("example", Example, lambda *_: None, ("harness self-test",))


@pytest.mark.parametrize("audio", [b"wrong", b"part"])
async def test_bad_or_missing_audio_is_a_server_failure(audio):
    with pytest.raises(AssertionError, match="server protocol failed"):
        async with STTServer(CASE, segments=(Segment(b"part-tail", "words"),)) as server:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(server.ws_url) as socket:
                    await socket.send_bytes(audio)
                    await socket.send_json({"type": "stop"})
                    async for _ in socket:
                        pass


async def test_unknown_stop_is_not_silently_accepted():
    with pytest.raises(AssertionError, match="unknown|server protocol failed"):
        async with STTServer(CASE) as server:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(server.ws_url) as socket:
                    await socket.send_json({"type": "not-a-stop"})
                    async for _ in socket:
                        pass


async def test_held_final_does_not_prevent_pongs_and_disconnect():
    async with STTServer(CASE, segments=(Segment(b"audio", "words"),)) as server:
        server.release_results.clear()
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(server.ws_url, autoping=False) as socket:
                await socket.send_bytes(b"audio")
                await socket.send_json({"type": "stop"})
                await asyncio.wait_for(server.input_ended.wait(), 1)
                await socket.ping(b"probe")
                while (
                    message := await asyncio.wait_for(socket.receive(), 1)
                ).type != aiohttp.WSMsgType.PONG:
                    assert not message.json()["final"]
                assert message.data == b"probe"
                assert server.connections[0].pings == 1
        await asyncio.wait_for(server.connections[0].disconnected.wait(), 1)
        server.check()


async def test_final_is_sent_only_after_release():
    async with STTServer(CASE, segments=(Segment(b"audio", "words"),)) as server:
        server.release_results.clear()
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(server.ws_url) as socket:
                await socket.send_bytes(b"audio")
                await socket.send_json({"type": "stop"})
                await server.input_ended.wait()
                server.release_results.set()
                messages = [message.json() async for message in socket]
                assert messages[-1] == {"text": "words", "final": True}


async def test_websockets_client_close_finishes_the_tcp_connection():
    import websockets

    async with STTServer(CASE, segments=(Segment(b"audio", "words"),)) as server:
        server.release_results.clear()
        socket = await websockets.connect(server.ws_url)
        try:
            await socket.send(b"audio")
            await socket.send('{"type":"stop"}')
            await asyncio.wait_for(server.input_ended.wait(), 1)
            await asyncio.wait_for(socket.close(), 1)
            await asyncio.wait_for(server.connections[0].disconnected.wait(), 1)
        finally:
            await socket.close()


async def test_writer_failure_cannot_look_like_a_successful_client_failure():
    with pytest.raises(AssertionError, match="server protocol failed"):
        async with STTServer(CASE) as server:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(server.ws_url) as socket:
                    await server.connected.wait()
                    server.connections[0].send({"not_json": object()})
                    async for _ in socket:
                        pass
