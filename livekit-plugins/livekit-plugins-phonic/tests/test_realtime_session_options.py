from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import llm, utils
from livekit.plugins.phonic.realtime import realtime_model
from livekit.plugins.phonic.realtime.realtime_model import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


class _Socket:
    def __init__(self, sent_configs: list[Any]) -> None:
        self._sent_configs = sent_configs
        self._closed = asyncio.Event()

    async def send_config(self, config: Any) -> None:
        self._sent_configs.append(config)

    async def send_audio_chunk(self, payload: Any) -> None:
        pass

    def __aiter__(self) -> _Socket:
        return self

    async def __anext__(self) -> Any:
        await self._closed.wait()
        raise StopAsyncIteration


class _SocketContext:
    def __init__(self, socket: _Socket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _Socket:
        return self._socket

    async def __aexit__(self, *args: object) -> None:
        self._socket._closed.set()


@pytest.fixture
def sent_configs(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    configs: list[Any] = []

    class _Conversations:
        def connect(self, **kwargs: object) -> _SocketContext:
            return _SocketContext(_Socket(configs))

    class _Phonic:
        def __init__(self, **kwargs: object) -> None:
            self.conversations = _Conversations()

    monkeypatch.setattr(realtime_model, "AsyncPhonic", _Phonic)
    return configs


@llm.function_tool(name="lookup", description="look something up")
async def _lookup() -> str:
    return "ok"


async def _wait_for_configs(configs: list[Any], count: int) -> None:
    for _ in range(100):
        if len(configs) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {count} configs, got {len(configs)}")


async def test_sends_seeded_config(sent_configs: list[Any]) -> None:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi there")
    session = RealtimeModel(api_key="test-key").session(
        instructions="be brief", chat_ctx=chat_ctx, tools=[_lookup]
    )
    try:
        await _wait_for_configs(sent_configs, 1)
        config = sent_configs[0]
        assert "be brief" in config.system_prompt
        assert "hi there" in config.system_prompt
        assert "lookup" in str(config.tools)
    finally:
        await session.aclose()


async def test_seeded_config_does_not_wait_for_omitted_tools(sent_configs: list[Any]) -> None:
    session = RealtimeModel(api_key="test-key").session(instructions="be brief")
    try:
        await _wait_for_configs(sent_configs, 1)
        assert sent_configs[0].system_prompt == "be brief"
    finally:
        await session.aclose()


async def test_seeded_config_waits_for_omitted_instructions(sent_configs: list[Any]) -> None:
    session = RealtimeModel(api_key="test-key").session(tools=[_lookup])
    try:
        await asyncio.sleep(0.02)
        assert not sent_configs
        await session.update_instructions("be brief")
        await _wait_for_configs(sent_configs, 1)
        assert "lookup" in str(sent_configs[0].tools)
    finally:
        await session.aclose()


async def test_instructions_are_per_session(sent_configs: list[Any]) -> None:
    model = RealtimeModel(api_key="test-key")
    first = model.session(instructions="first")
    second = model.session(instructions="second")
    third = model.session(tools=[_lookup])
    try:
        await _wait_for_configs(sent_configs, 2)
        await asyncio.sleep(0.02)
        prompts = sorted(config.system_prompt for config in sent_configs)
        assert prompts == ["first", "second"]
        assert not utils.is_given(model._opts.instructions)
    finally:
        await asyncio.gather(first.aclose(), second.aclose(), third.aclose())


async def test_session_without_options_waits_for_framework(sent_configs: list[Any]) -> None:
    session: RealtimeSession = RealtimeModel(api_key="test-key").session()
    try:
        await asyncio.sleep(0.02)
        assert not sent_configs
        await session._update_session(
            instructions="be brief", chat_ctx=llm.ChatContext.empty(), tools=[]
        )
        await _wait_for_configs(sent_configs, 1)
    finally:
        await session.aclose()
