"""Phonic realtime plugin: user text and tool turns through a local conversation socket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, llm
from livekit.plugins.phonic.realtime import RealtimeModel, RealtimeSession
from tests.fakes import until

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


class _Peer:
    def __init__(self) -> None:
        self.socket: asyncio.Future[web.WebSocketResponse] = (
            asyncio.get_running_loop().create_future()
        )
        self.received: asyncio.Queue[Wire] = asyncio.Queue()

    async def conversation(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.socket.set_result(socket)
        async for message in socket:
            if message.type == web.WSMsgType.TEXT:
                self.received.put_nowait(json.loads(message.data))
        return socket

    async def next(self, kind: str) -> Wire:
        while True:
            event = await asyncio.wait_for(self.received.get(), 1)
            if event["type"] == kind:
                return event

    async def send(self, event: dict[str, object]) -> None:
        await (await self.socket).send_json(event)


@llm.function_tool
async def lookup(place: str) -> str:
    """Look a place up."""
    return place


@llm.function_tool
async def hang_up() -> str:
    """End the call."""
    return ""


@pytest.fixture
async def conversation() -> AsyncIterator[tuple[_Peer, RealtimeSession, Wire]]:
    peer = _Peer()
    app = web.Application()
    app.router.add_get("/v1/sts/ws", peer.conversation)
    async with TestServer(app) as server:
        model = RealtimeModel(
            api_key="key",
            conn_options=APIConnectOptions(max_retry=0),
            base_url=str(server.make_url("").with_scheme("ws")),
            default_language="en",
            additional_languages=["es"],
            configs_for_tools=[{"name": "hang_up", "forbid_speech_after_tool_call": True}],
        )
        session = model.session()
        try:
            await session.update_instructions("Be brief.")
            await session.update_tools([lookup, hang_up])
            config = await peer.next("config")
            await peer.send({"type": "ready_to_start_conversation"})
            yield peer, session, config
        finally:
            await session.aclose()


async def test_user_text_reaches_the_next_reply_exactly_once(
    conversation: tuple[_Peer, RealtimeSession, Wire],
) -> None:
    """Typed user text is sent with the reply it asks for, and never again.

    https://github.com/livekit/agents/pull/6169
    """
    peer, session, _ = conversation
    chat_ctx = session.chat_ctx
    chat_ctx.add_message(role="user", content="Which shops are open late?")
    await session.update_chat_ctx(chat_ctx)

    replies = [session.generate_reply()]
    first = await peer.next("generate_reply")
    replies.append(session.generate_reply())
    second = await peer.next("generate_reply")
    for reply in replies:
        reply.cancel()
    assert first["system_message"].count("Which shops are open late?") == 1
    assert "Which shops are open late?" not in (second.get("system_message") or "")


async def test_a_speech_forbidding_tool_opens_no_reply_after_its_output(
    conversation: tuple[_Peer, RealtimeSession, Wire],
) -> None:
    """Only the configured tool forbids speech; its output opens no assistant turn, others do.

    https://github.com/livekit/agents/pull/6143 https://github.com/livekit/agents/pull/5285
    """
    peer, session, config = conversation
    assert config["default_language"] == "en"
    assert config["additional_languages"] == ["es"]
    forbids = {
        t["tool_schema"]["function"]["name"]: t["forbid_speech_after_tool_call"]
        for t in config["tools"]
    }
    assert forbids == {"lookup": False, "hang_up": True}

    turns: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", turns.append)

    async def answer(session: RealtimeSession, call_id: str, name: str) -> int:
        """Turns the tool's output opens."""
        before = len(turns)
        await peer.send(
            {"type": "tool_call", "tool_call_id": call_id, "tool_name": name, "parameters": {}}
        )
        # the call arrives on a turn of its own
        await until(lambda: len(turns) > before, f"{name} arrived on a turn")
        before = len(turns)
        chat_ctx = session.chat_ctx
        chat_ctx.insert(llm.FunctionCall(call_id=call_id, name=name, arguments="{}"))
        chat_ctx.insert(
            llm.FunctionCallOutput(call_id=call_id, name=name, output="done", is_error=False)
        )
        await session.update_chat_ctx(chat_ctx)
        sent = await peer.next("tool_call_output")
        assert sent["tool_call_id"] == call_id
        return len(turns) - before

    assert await answer(session, "t1", "hang_up") == 0
    assert await answer(session, "t2", "lookup") == 1


async def test_a_provider_validation_error_settles_the_pending_reply_and_keeps_its_code(
    conversation: tuple[_Peer, RealtimeSession, Wire],
) -> None:
    """Phonic's observed validation envelope must fail the waiter and preserve diagnostics."""
    peer, session, config = conversation
    assert "event_id" not in config  # Phonic rejects OpenAI's extra configuration field.
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    reply = session.generate_reply()
    await peer.next("generate_reply")
    await peer.send(
        {
            "type": "error",
            "error": {"code": "user_message_validation_failed", "message": "Invalid parameters"},
            "param_errors": {"type": "Invalid discriminator value"},
        }
    )
    await until(lambda: reply.done() and bool(errors), "validation settles reply and error")
    assert reply.exception() is not None
    assert not errors[-1].recoverable
    error = errors[-1].error
    assert isinstance(error, APIError)
    assert error.kind == "invalid_request"
    assert error.detail is not None
    assert json.loads(error.detail)["error"]["code"] == "user_message_validation_failed"


async def test_reply_timeout_and_old_callbacks_do_not_settle_the_successor(
    conversation: tuple[_Peer, RealtimeSession, Wire],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.plugins.reply_timers import capture_reply_timers

    peer, session, _ = conversation
    timers = capture_reply_timers(monkeypatch, 10.0)
    first = session.generate_reply()
    await peer.next("generate_reply")
    assert len(timers) == 1
    timers[0].fire()
    await until(first.done, "the first reply times out")
    with pytest.raises(llm.RealtimeError, match="timed out"):
        first.result()
    second = session.generate_reply()
    await peer.next("generate_reply")
    assert len(timers) == 2
    timers[0].fire(stale=True)
    assert not second.done()
    await peer.send({"type": "assistant_started_speaking"})
    await until(second.done, "the next reply is correlated")
    generation = second.result()
    assert generation.user_initiated
    timers[1].fire(stale=True)
    assert second.result() is generation
    await peer.send({"type": "assistant_finished_speaking"})
