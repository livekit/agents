"""Mistral AI plugin: LLM request format and errors, STT, through the real SDK and a local server."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx as httpx1
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from mistralai.client import Mistral

from livekit import rtc
from livekit.agents import Agent, APIConnectOptions, APIError, APIStatusError, RunContext, llm
from livekit.agents.llm import ChatContext
from livekit.plugins.mistralai import STT
from livekit.plugins.mistralai.llm import LLM, ApiMode
from tests_new.fakes import until

RETRIES = APIConnectOptions(max_retry=3, retry_interval=0.0)


# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


def _conversation_sse(*events: Wire) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _completion_sse(*deltas: Wire, finish_reason: str | None = None) -> bytes:
    chunks = [
        {
            "id": "cmpl_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason if i == len(deltas) - 1 else None,
                }
            ],
        }
        for i, delta in enumerate(deltas)
    ]
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


_DONE = {
    "type": "conversation.response.done",
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


@contextlib.asynccontextmanager
async def _mistral_server(
    replies: list[bytes | web.Response | tuple[bytes, str]],
) -> AsyncIterator[tuple[Mistral, list[tuple[str, Wire]]]]:
    """A local Mistral API answering each request with the next reply, via the real SDK.

    A reply is an SSE body, a ``web.Response``, or ``(sse_body, "drop")`` to drop the connection
    after the body.
    """
    received: list[tuple[str, Wire]] = []

    async def answer(request: web.Request) -> web.StreamResponse:
        if request.content_type == "application/json":
            received.append((request.path, await request.json()))
        else:
            received.append((request.path, dict(await request.post())))
        reply = replies[min(len(received), len(replies)) - 1]
        if isinstance(reply, web.Response):
            return reply
        if isinstance(reply, tuple):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(reply[0])
            assert request.transport is not None
            request.transport.close()
            return response
        return web.Response(body=reply, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/{path:.*}", answer)
    async with TestServer(app) as server:
        client = Mistral(api_key="test", server_url=str(server.make_url("")).rstrip("/"))
        try:
            yield client, received
        finally:
            await client.__aexit__(None, None, None)
            client.__exit__(None, None, None)


async def _reply(
    model: LLM, chat_ctx: ChatContext | None = None, *, conn_options: APIConnectOptions
) -> list[llm.ChatChunk]:
    chunks: list[llm.ChatChunk] = []
    try:
        async with model.chat(
            chat_ctx=chat_ctx or ChatContext.empty(), conn_options=conn_options
        ) as stream:
            async for chunk in stream:
                chunks.append(chunk)
    finally:
        await model.aclose()
    return chunks


async def test_chat_mode_sends_developer_instructions_as_system_messages() -> None:
    """In chat-completions mode a developer message is sent as a system message."""
    context = ChatContext.empty()
    context.add_message(role="developer", content="Follow these instructions.")
    context.add_message(role="user", content="Hello.")
    async with _mistral_server([_completion_sse({"content": "Hi."})]) as (client, received):
        model = LLM(client=client, model="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        await _reply(model, context, conn_options=APIConnectOptions(max_retry=0))

    [(path, body)] = received
    assert path == "/v1/chat/completions"
    assert body["messages"] == [
        {"role": "system", "content": "Follow these instructions."},
        {"role": "user", "content": "Hello."},
    ]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, '{"message":"provider error"}'),
        (422, '{"detail":[{"loc":["body"],"msg":"invalid request","type":"value_error"}]}'),
    ],
)
async def test_client_error_is_not_retried(status: int, body: str) -> None:
    """A 4xx keeps its status, request id and body, and is not retried.

    https://github.com/livekit/agents/pull/7308
    """
    refusal = web.Response(
        status=status, text=body, content_type="application/json", headers={"x-request-id": "req_1"}
    )
    async with _mistral_server([refusal]) as (client, received):
        model = LLM(client=client, model="test")
        errors: list[llm.LLMError] = []
        model.on("error", errors.append)

        with pytest.raises(APIStatusError) as caught:
            await _reply(model, conn_options=RETRIES)

    assert len(received) == 1
    assert (caught.value.status_code, caught.value.request_id) == (status, "req_1")
    assert caught.value.body == body
    assert caught.value.retryable is False
    assert [(e.error, e.recoverable) for e in errors] == [(caught.value, False)]


async def test_failure_after_output_is_not_retried() -> None:
    """A stream that fails after some text keeps that text and is not replayed.

    https://github.com/livekit/agents/pull/7308
    """
    partial = _conversation_sse(
        {"type": "message.output.delta", "id": "msg_1", "content": "partial"}
    )
    async with _mistral_server([(partial, "drop")]) as (client, received):
        model = LLM(client=client, model="test")
        chunks: list[llm.ChatChunk] = []

        with pytest.raises(APIError) as caught:
            async with model.chat(chat_ctx=ChatContext.empty(), conn_options=RETRIES) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
        await model.aclose()

    assert [chunk.delta.content for chunk in chunks if chunk.delta] == ["partial"]
    assert len(received) == 1
    assert caught.value.retryable is False


async def test_tool_call_accumulation() -> None:
    """Streamed tool-call argument fragments become one complete call.

    https://github.com/livekit/agents/pull/7381
    """
    first = {
        "tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": '{"loc'}}
        ]
    }
    second = {
        "tool_calls": [{"index": 0, "function": {"name": "", "arguments": 'ation": "paris"}'}}]
    }
    reply = _completion_sse(first, second, finish_reason="tool_calls")
    async with _mistral_server([reply]) as (client, _):
        model = LLM(client=client, model="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        chunks = await _reply(model, conn_options=APIConnectOptions(max_retry=0))

    [call] = [c for chunk in chunks if chunk.delta for c in chunk.delta.tool_calls]
    assert (call.call_id, call.name, call.arguments) == (
        "call_1",
        "get_weather",
        '{"location": "paris"}',
    )


async def test_transcription_without_segment_timestamps_keeps_text() -> None:
    """Segments without times still give the text, with unknown word times."""
    transcription = {
        "model": "voxtral-mini-latest",
        "text": "hello",
        "language": "en",
        "usage": {},
        "segments": [{"text": "hello", "start": None, "end": None}],
    }
    async with _mistral_server([web.json_response(transcription)]) as (client, _):
        model = STT(client=client)
        frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)

        result = await model.recognize(frame, conn_options=APIConnectOptions(max_retry=0))
        await model.aclose()

    [alternative] = result.alternatives
    assert alternative.text == "hello"
    assert alternative.words is not None
    assert (alternative.words[0].start_time, alternative.words[0].end_time) == (None, None)


async def test_sdk_stream_assembles_text_and_tool_calls_and_replays_full_history() -> None:
    """The installed SDK's wire shapes round-trip, statelessly; https://github.com/livekit/agents/pull/3157 https://github.com/livekit/agents/issues/5115 https://github.com/livekit/agents/pull/5586."""
    call = {"type": "function.call.delta", "id": "fc_1", "name": "lookup", "tool_call_id": "call_1"}
    first = _conversation_sse(
        {"type": "message.output.delta", "id": "msg_1", "content": "Checking."},
        {**call, "arguments": '{"room": '},
        {**call, "arguments": '"203"}'},
        _DONE,
    )
    second = _conversation_sse(
        {"type": "message.output.delta", "id": "msg_2", "content": "Room 203 is free."}, _DONE
    )
    async with _mistral_server([first, second]) as (client, received):
        model = LLM(client=client, model="mistral-small-latest")
        agent = Agent(instructions="Check rooms.", llm=model)

        @agent.tool
        def lookup(ctx: RunContext, room: str) -> str:
            """Check whether a room is free."""
            return f"{room} is free"

        result = await agent.run("Is room 203 free?")

    assert result.status == "completed"

    assert result.output == "Room 203 is free."
    assert [path for path, _ in received] == ["/v1/conversations", "/v1/conversations"]
    body = received[1][1]
    assert "conversation_id" not in body
    assert body["instructions"] == "Check rooms."
    assert [
        (e["type"], e.get("content") or e.get("arguments") or e.get("result"))
        for e in body["inputs"]
    ] == [
        ("message.input", "Is room 203 free?"),
        ("message.output", "Checking."),
        ("function.call", '{"room": "203"}'),
        ("function.result", "203 is free"),
    ]
    assert {e.get("tool_call_id") for e in body["inputs"][2:]} == {"call_1"}


async def test_base_prompt_and_assistant_ending_context_are_sent_validly() -> None:
    """Keep the base prompt, inline later instructions, end on a user entry; https://github.com/livekit/agents/pull/6591 https://github.com/livekit/agents/issues/7030 https://github.com/livekit/agents/pull/7036."""
    reply = _conversation_sse(
        {"type": "message.output.delta", "id": "msg_1", "content": "Sure."}, _DONE
    )
    context = ChatContext.empty()
    context.add_message(role="system", content="You are the front desk.")
    context.add_message(role="user", content="Hi.")
    context.add_message(role="system", content="Mention the breakfast hours.")
    context.add_message(role="assistant", content="Hello!")
    async with _mistral_server([reply]) as (client, received):
        model = LLM(client=client, model="mistral-small-latest")
        async with model.chat(chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)) as s:
            text = "".join([c.delta.content async for c in s if c.delta and c.delta.content])

    assert text == "Sure."
    body = received[0][1]
    assert body["instructions"] == "You are the front desk."
    assert [(e["type"], e["content"]) for e in body["inputs"]] == [
        ("message.input", "Hi."),
        ("message.input", "<instructions>\nMention the breakfast hours.\n</instructions>"),
        ("message.output", "Hello!"),
        ("message.input", "(empty)"),
    ]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/pull/8169 reproduces: in-band errors are retried",
)
async def test_in_band_error_is_not_retried() -> None:
    """A provider error event fails the request once. https://github.com/livekit/agents/pull/8169"""
    error = _conversation_sse(
        {"type": "conversation.response.error", "message": "model overloaded", "code": 3000}
    )
    async with _mistral_server([error]) as (client, received):
        model = LLM(client=client, model="test")

        with pytest.raises(APIError) as caught:
            await _reply(model, conn_options=RETRIES)

    assert len(received) == 1
    assert isinstance(caught.value, APIStatusError)
    assert caught.value.retryable is False


@pytest.mark.parametrize("language", ["en", None])
async def test_transcription_never_combines_language_with_timestamps(language: str | None) -> None:
    """Mistral returns empty text when both are set; https://github.com/livekit/agents/issues/4399."""
    sent: list[tuple[object, object]] = []

    async def transcribe(request: web.Request) -> web.Response:
        fields = await request.post()
        sent.append((fields.get("language"), fields.getall("timestamp_granularities", [])))
        return web.json_response(
            {
                "model": "voxtral-mini-latest",
                "text": "hello there",
                "language": None,
                "usage": {},
                "segments": [{"text": "hello there", "start": 0.0, "end": 0.5}],
            }
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", transcribe)
    frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
    async with TestServer(app) as server:
        client = Mistral(api_key="test", server_url=str(server.make_url("")).rstrip("/"))
        model = STT(client=client, language=language) if language else STT(client=client)
        try:
            result = await model.recognize(frame, conn_options=APIConnectOptions(max_retry=0))
        finally:
            await model.aclose()
            await client.__aexit__(None, None, None)
            client.__exit__(None, None, None)

    assert result.alternatives[0].text == "hello there"
    assert sent == [(language, [] if language else ["segment"])]


async def test_realtime_transcription_handshake_carries_client_headers() -> None:
    """Custom client headers reach the realtime socket; https://github.com/livekit/agents/pull/5575."""
    handshakes: list[Mapping[str, str]] = []

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        handshakes.append(request.headers.copy())
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        session = {
            "request_id": "req_1",
            "model": "voxtral-mini-transcribe-realtime-2602",
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
        }
        await ws.send_json({"type": "session.created", "session": session})
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/v1/audio/transcriptions/realtime", realtime)
    async with TestServer(app) as server:
        client = Mistral(
            api_key="test",
            server_url=str(server.make_url("")).rstrip("/"),
            async_client=httpx1.AsyncClient(headers={"X-Proxy-Token": "proxy-secret"}),
        )
        model = STT(client=client, model="voxtral-mini-transcribe-realtime-2602")
        try:
            async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_frame(
                    rtc.AudioFrame.create(
                        sample_rate=16000, num_channels=1, samples_per_channel=800
                    )
                )
                await until(lambda: handshakes, "the realtime socket opens")
        finally:
            await model.aclose()
            await client.__aexit__(None, None, None)
            client.__exit__(None, None, None)

    assert len(handshakes) == 1
    assert handshakes[0].get("X-Proxy-Token") == "proxy-secret"
    assert handshakes[0].get("Authorization") == "Bearer test"
