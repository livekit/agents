from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import Agent, AgentSession, AgentTask, inference
from livekit.agents.tts import FallbackAdapter

from .fake_llm import FakeLLM
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit]


class _Gateway:
    """Local stand-in for the inference gateway that tracks open TTS websockets."""

    def __init__(self) -> None:
        self.open: set[web.WebSocketResponse] = set()
        self.opened = 0

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.open.add(ws)
        self.opened += 1
        try:
            async for _ in ws:
                pass
        finally:
            self.open.discard(ws)
        return ws


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[_Gateway, str]]:
    gw = _Gateway()
    app = web.Application()
    app.router.add_get("/tts", gw.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield gw, f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def _settled(cond, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


async def test_handed_off_agent_tts_closes_its_gateway_socket(
    gateway: tuple[_Gateway, str],
) -> None:
    gw, base_url = gateway
    async with aiohttp.ClientSession() as http:

        def make_tts() -> inference.TTS:
            return inference.TTS(
                model="cartesia/sonic-3",
                api_key="k",
                api_secret="s",
                base_url=base_url,
                http_session=http,
            )

        agents = [Agent(instructions=n, tts=make_tts()) for n in "abc"]
        session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
        await session.start(agents[0])
        try:
            await _settled(lambda: gw.opened == 1)

            for n, agent in enumerate(agents[1:], start=2):
                session.update_agent(agent)
                await _settled(lambda n=n: gw.opened == n)  # type: ignore[misc]
                await asyncio.sleep(0.2)
                # only the current agent's TTS holds a socket
                assert len(gw.open) == 1
        finally:
            await session.aclose()

        await _settled(lambda: len(gw.open) == 0)


class _CountingTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__()
        self.released = 0

    async def release_idle_connections(self) -> None:
        self.released += 1


def _session(**kwargs) -> AgentSession:  # type: ignore[no-untyped-def]
    return AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": None}, **kwargs)


async def _handoff(session: AgentSession, agent: Agent) -> None:
    session.update_agent(agent)
    await _settled(lambda: session._activity is not None and session._activity.agent is agent)


async def test_releases_displaced_agent_tts_but_never_the_session_tts() -> None:
    session_tts, tts_a, tts_b = _CountingTTS(), _CountingTTS(), _CountingTTS()
    session = _session(tts=session_tts)
    await session.start(Agent(instructions="a", tts=tts_a))
    await _handoff(session, Agent(instructions="b", tts=tts_b))
    await _handoff(session, Agent(instructions="c"))
    await session.aclose()

    assert (tts_a.released, tts_b.released, session_tts.released) == (1, 1, 0)


async def test_keeps_an_agent_tts_the_next_agent_also_uses() -> None:
    shared = _CountingTTS()
    session = _session()
    await session.start(Agent(instructions="a", tts=shared))
    await _handoff(session, Agent(instructions="b", tts=shared))
    assert shared.released == 0

    await session.aclose()
    assert shared.released == 1


async def test_releases_a_task_tts_on_completion_not_the_paused_parent_tts() -> None:
    parent_tts, task_tts = _CountingTTS(), _CountingTTS()
    task_done = asyncio.Event()

    class Task(AgentTask[None]):
        async def on_enter(self) -> None:
            self.complete(None)

    class Parent(Agent):
        async def on_enter(self) -> None:
            await Task(instructions="task", tts=task_tts)
            task_done.set()

    session = _session()
    await session.start(Parent(instructions="parent", tts=parent_tts))
    await asyncio.wait_for(task_done.wait(), 2.0)
    assert (task_tts.released, parent_tts.released) == (1, 0)

    await session.aclose()
    assert parent_tts.released == 1


async def test_releases_every_provider_behind_an_agent_fallback_adapter() -> None:
    primary, secondary = _CountingTTS(), _CountingTTS()
    adapter = FallbackAdapter([primary, secondary])
    session = _session()
    await session.start(Agent(instructions="a", tts=adapter))
    await _handoff(session, Agent(instructions="b"))
    await session.aclose()
    await adapter.aclose()

    assert (primary.released, secondary.released) == (1, 1)
