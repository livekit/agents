"""Host objects the core calls back are freed once dropped: the core holds a back-reference to
an object that owns it weakly, never as a cycle the collector cannot see."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from livekit.agents import Agent, TextSession, llm
from livekit.agents.llm.mcp import MCPServerHTTP, MCPToolset
from livekit.agents.voice.avatar import DataStreamAudioReceiver
from tests.fakes import AgentRoom, FakeLLM, FakeSTT, FakeTTS, audio_frame, until

from .mcp_peer import serve_http


async def _llm_stream() -> object:
    stream = FakeLLM(["Hi."]).chat(chat_ctx=llm.ChatContext.empty())
    async with stream:
        async for _ in stream:
            pass
    return stream


async def _recognize_stream() -> object:
    stream = FakeSTT(["hi"]).stream()
    stream.push_frame(audio_frame())
    stream.end_input()
    async for _ in stream:
        pass
    await stream.aclose()
    return stream


async def _recognition() -> object:
    model = FakeSTT(["hi"])
    await model.recognize([audio_frame()])
    return model


async def _chunked_stream() -> object:
    stream = FakeTTS().synthesize("hello")
    async with stream:
        async for _ in stream:
            pass
    return stream


async def _synthesize_stream() -> object:
    stream = FakeTTS().stream()
    stream.push_text("hello")
    stream.end_input()
    async for _ in stream:
        pass
    await stream.aclose()
    return stream


async def _audio_receiver() -> object:
    receiver = DataStreamAudioReceiver(AgentRoom())
    await receiver.aclose()
    return receiver


STREAMS: dict[str, Callable[[], Awaitable[object]]] = {
    "LLMStream": _llm_stream,
    "RecognizeStream": _recognize_stream,
    # a recognition's request is the STT's own: the STT is what it would keep
    "STT.recognize": _recognition,
    "ChunkedStream": _chunked_stream,
    "SynthesizeStream": _synthesize_stream,
    "DataStreamAudioReceiver": _audio_receiver,
}


@pytest.mark.parametrize("name", sorted(STREAMS))
async def test_a_finished_provider_request_is_freed(name: str) -> None:
    """A request's attempts and a receiver's clear listener call back their own host object,
    which owns them."""
    freed = weakref.ref(await STREAMS[name]())
    await until(lambda: gc.collect() >= 0 and freed() is None, f"a finished {name} is freed")


async def test_a_dropped_mcp_toolset_waiting_on_its_idle_close_is_freed(tmp_path: Path) -> None:
    """A run releases the toolset, arming its idle close, whose lifecycle calls it back."""

    async def used() -> weakref.ref[MCPToolset]:
        async with serve_http(tmp_path / "peer") as peer:
            toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))
            agent = Agent(instructions="Help.", llm=FakeLLM(["Hello."]), tools=[toolset])
            await TextSession(agent=agent).run("hi")
            return weakref.ref(toolset)

    freed = await used()
    await until(lambda: gc.collect() >= 0 and freed() is None, "a dropped MCP toolset is freed")
