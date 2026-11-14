"""The inference LLM keeps a spare connection for the request that follows a cancelled one.

The client speaks HTTP/1.1 to the gateway, where a cancelled request closes its
connection. Preemptive generation cancels a request and sends its replacement at once,
so without a spare the replacement opens a new connection before it can start.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import openai
import pytest

import livekit.agents.inference.llm as inference_llm
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, llm
from livekit.agents.inference import LLM

pytestmark = pytest.mark.unit

_CHAT_PATH = "/v1/chat/completions"
_TEXT_CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "google/gemma-4-31b-it",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}}],
}


def _http_chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class _Gateway:
    """A loopback HTTP/1.1 server that records which connection served each request."""

    def __init__(self) -> None:
        self.connections = 0
        self.served: list[tuple[str, int]] = []
        # a stalled chat request sends one chunk, then waits for the client to hang up
        self.stall_chat = False
        self._unanswered = 0
        self._server: asyncio.Server | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    def chat_connections(self) -> list[int]:
        return [connection for path, connection in self.served if path == _CHAT_PATH]

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)

    async def aclose(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def wait_idle(self, *, connections: int) -> None:
        """Wait until ``connections`` have been opened and every request is answered."""

        async def _poll() -> None:
            while self.connections < connections or self._unanswered:
                await asyncio.sleep(0.005)

        try:
            await asyncio.wait_for(_poll(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail(f"expected {connections} connections, the gateway saw {self.connections}")

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        connection = self.connections
        try:
            while request_line := await reader.readline():
                path = request_line.split()[1].decode()
                body_len = 0
                while (header := await reader.readline()) not in (b"\r\n", b""):
                    name, _, value = header.partition(b":")
                    if name.lower() == b"content-length":
                        body_len = int(value)
                await reader.readexactly(body_len)

                self._unanswered += 1
                try:
                    self.served.append((path, connection))
                    if path == _CHAT_PATH:
                        await self._answer_chat(reader, writer)
                    else:
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                            b"Content-Length: 2\r\n\r\n{}"
                        )
                        await writer.drain()
                finally:
                    self._unanswered -= 1
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _answer_chat(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            + _http_chunk(f"data: {json.dumps(_TEXT_CHUNK)}\n\n".encode())
        )
        await writer.drain()
        if self.stall_chat:
            await reader.read()  # returns once the client closes the connection
            raise ConnectionResetError
        writer.write(_http_chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")
        await writer.drain()


@pytest.fixture
async def gateway() -> AsyncIterator[_Gateway]:
    gateway = _Gateway()
    await gateway.start()
    yield gateway
    await gateway.aclose()


@pytest.fixture
async def gateway_llm(gateway: _Gateway) -> AsyncIterator[LLM]:
    # Long enough to keep PyJWT's short-key warning out of the suite output.
    fake_secret = "f" * 32
    gateway_llm = LLM(
        model="google/gemma-4-31b-it",
        base_url=gateway.base_url,
        api_key=fake_secret,
        api_secret=fake_secret,
    )
    gateway_llm.prewarm()
    assert gateway_llm._prewarm_task is not None
    await gateway_llm._prewarm_task
    yield gateway_llm
    await gateway_llm.aclose()


def _chat(llm_model: llm.LLM) -> llm.LLMStream:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    return llm_model.chat(
        chat_ctx=chat_ctx, conn_options=APIConnectOptions(max_retry=0, timeout=5.0)
    )


async def _cancel_mid_stream(llm_model: llm.LLM, gateway: _Gateway) -> llm.LLMStream:
    gateway.stall_chat = True
    stream = _chat(llm_model)
    await stream.__anext__()  # the request is in flight and holds a connection
    await stream.aclose()
    gateway.stall_chat = False
    return stream


async def _complete(llm_model: llm.LLM) -> None:
    async with _chat(llm_model) as stream:
        async for _ in stream:
            pass


def _hold_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a pending replacement connection out of the way of the request under test."""
    monkeypatch.setattr(inference_llm, "_SPARE_CONNECTION_DELAY", 1.0)


async def test_request_after_a_cancelled_one_uses_an_open_connection(
    gateway_llm: LLM, gateway: _Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hold_replacement(monkeypatch)
    await _cancel_mid_stream(gateway_llm, gateway)
    open_before = gateway.connections

    await _complete(gateway_llm)

    assert gateway.chat_connections()[-1] <= open_before, (
        "the request that followed a cancelled one had to open a new connection"
    )


async def test_spare_connection_is_replaced_after_a_cancel(
    gateway_llm: LLM, gateway: _Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _cancel_mid_stream(gateway_llm, gateway)
    # the cancelled request closed one connection: its replacement opens in the background
    await gateway.wait_idle(connections=gateway.connections + 1)

    _hold_replacement(monkeypatch)
    await _cancel_mid_stream(gateway_llm, gateway)
    open_before = gateway.connections
    await _complete(gateway_llm)

    assert gateway.chat_connections()[-1] <= open_before, (
        "after two cancelled requests the next one had to open a new connection"
    )


class _ProviderLLM(llm.LLM):
    """An LLM for another provider that reuses the inference stream, as plugins do."""

    def __init__(self, client: openai.AsyncClient) -> None:
        super().__init__()
        self._client = client

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_: Any,
    ) -> llm.LLMStream:
        return inference_llm.LLMStream(
            self,
            model="provider-model",
            strict_tool_schema=True,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs={},
        )


async def test_cancelled_request_to_another_provider_sends_nothing_more(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inference_llm, "_SPARE_CONNECTION_DELAY", 0.0)
    client = openai.AsyncClient(api_key="test-key", base_url=gateway.base_url, max_retries=0)
    try:
        stream = await _cancel_mid_stream(_ProviderLLM(client), gateway)
        await asyncio.sleep(0.05)
    finally:
        await client.close()

    assert stream._task.cancelled(), "the cancelled request ended with an error"
    assert gateway.served == [(_CHAT_PATH, 1)]
