from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import pytest

from livekit.agents import llm
from livekit.plugins.openai.realtime.gpt_live_model import (
    GPTLiveDelegation,
    GPTLiveModel,
    GPTLiveSession,
)
from livekit.plugins.openai.tools import WebSearch

pytestmark = pytest.mark.unit


class _FakeWS:
    """A websocket that accepts everything and never delivers anything."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self) -> None:
        await asyncio.Event().wait()

    async def close(self) -> None:
        pass


def _connect_hook(monkeypatch: pytest.MonkeyPatch) -> _FakeWS:
    """Replace the handshake before any session exists, so nothing can reach the network."""
    ws = _FakeWS()

    async def _create_ws_conn(self: GPTLiveSession) -> _FakeWS:
        return ws

    monkeypatch.setattr(GPTLiveSession, "_create_ws_conn", _create_ws_conn)
    return ws


def _chat_ctx() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="a prior turn", id="m1")
    return ctx


@llm.function_tool
async def _get_weather(location: str) -> str:
    """Get the weather."""
    return "rainy"


async def test_a_session_awaiting_config_sends_nothing_until_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connection may win the race with the configuration, and must still wait for it."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session(wait_for_config=True)
    try:
        await asyncio.sleep(0.05)  # let the connection open well ahead of any configuration
        assert not ws.sent, "session.update went out before the configuration arrived"

        await session._update_session(
            instructions="Be concise.", chat_ctx=_chat_ctx(), tools=[_get_weather]
        )
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent] == ["session.update"]
        assert ws.sent[0]["session"]["instructions"] == "Be concise."
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_session_built_directly_starts_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody promised this one a configuration, so waiting for one would hang it."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", instructions="From the constructor.")
    session = model.session()
    try:
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent] == ["session.update"]
        assert ws.sent[0]["session"]["instructions"] == "From the constructor."
    finally:
        await session.aclose()
        await model.aclose()


async def test_closing_releases_a_session_still_waiting_for_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session closed before it was ever configured has to finish closing.

    Judged on elapsed time: ``aclose`` suppresses ``CancelledError``, so a deadline never fails.
    """
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session(wait_for_config=True)
    await asyncio.sleep(0.05)

    started = time.perf_counter()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(session.aclose(), timeout=2)  # bounds a real hang
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"aclose blocked {elapsed:.2f}s on a configuration that never came"
    await model.aclose()


async def test_client_delegation_reaches_the_application_and_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", delegation="client")
    session = model.session(wait_for_config=True)
    delegations: list[GPTLiveDelegation] = []
    session.on("delegation_created", delegations.append)
    try:
        await session._update_session(instructions="Be concise.", tools=[_get_weather])
        await asyncio.sleep(0.05)
        assert ws.sent[0]["session"]["delegation"] == {"type": "client"}

        # a responses-targeted delegation is the backend's, so it must not reach the app
        session._handle_event(
            {
                "type": "delegation.created",
                "item": {"id": "item_1", "type": "delegation", "target": "responses"},
            }
        )
        assert not delegations

        session._handle_event(
            {
                "type": "delegation.created",
                "offset_ms": 1000,
                "item": {
                    "id": "item_delegation_123",
                    "type": "delegation",
                    "target": "client",
                    "content": [{"type": "input_text", "text": "What is the weather?"}],
                },
            }
        )
        # a plugin type, not the wire event: the alpha's shape must not reach the application
        assert delegations == [
            GPTLiveDelegation(id="item_delegation_123", text="What is the weather?")
        ]

        session.send_delegation_context(delegation_id=delegations[0].id, text="62 and raining.")
        await asyncio.sleep(0.05)
        answer = ws.sent[-1]
        assert answer["type"] == "delegation.context.append"
        assert answer["delegation_item_id"] == "item_delegation_123"
        assert answer["channel"] == "speakable"
        assert answer["content"] == [{"type": "input_text", "text": "62 and raining."}]
    finally:
        await session.aclose()
        await model.aclose()


async def test_delegation_target_switches_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later session.update is sparse, so the switch carries delegation and nothing else."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", backend_model="gpt-5.6-sol")
    session = model.session(wait_for_config=True)
    try:
        await session._update_session(instructions="Be concise.", tools=[])
        await asyncio.sleep(0.05)
        assert ws.sent[0]["session"]["delegation"]["type"] == "responses"

        session.update_delegation("client")
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["session"] == {"delegation": {"type": "client"}}

        # switching back to responses has to carry the backend model again
        session.update_delegation("responses")
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["session"]["delegation"]["responses"]["model"] == "gpt-5.6-sol"
        assert "instructions" not in ws.sent[-1]["session"]
    finally:
        await session.aclose()
        await model.aclose()


async def test_first_event_is_a_session_update_carrying_the_whole_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", voice="marin")
    session = model.session(wait_for_config=True)
    try:
        await session._update_session(
            instructions="Be concise.", chat_ctx=_chat_ctx(), tools=[_get_weather]
        )
        await asyncio.sleep(0.1)  # let the connection open and the send loop drain

        assert ws.sent, "nothing reached the wire"
        first = ws.sent[0]
        assert first["type"] == "session.update"
        config = first["session"]
        assert config["instructions"] == "Be concise."
        assert config["audio"]["output"]["voice"] == "marin"
        assert [t["name"] for t in config["delegation"]["responses"]["tools"]] == ["_get_weather"]
        assert config["initial_items"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "a prior turn"}],
            }
        ]
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_hosted_tool_is_delegated_to_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenAI provider tool goes to the backend as its own entry, next to the function tools."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session(wait_for_config=True)
    try:
        await session._update_session(
            instructions="Be concise.",
            chat_ctx=llm.ChatContext.empty(),
            tools=[_get_weather, WebSearch(search_context_size="low")],
        )
        await asyncio.sleep(0.1)

        tools = ws.sent[0]["session"]["delegation"]["responses"]["tools"]
        assert tools[0]["name"] == "_get_weather"
        assert tools[1] == {"type": "web_search", "search_context_size": "low"}
    finally:
        await session.aclose()
        await model.aclose()
