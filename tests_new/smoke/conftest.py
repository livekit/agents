"""Smoke fixtures: a credential-free environment, the console's CLI end, agent modules run as
``python agent.py <command>``, and the examples loaded from their own folders."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import struct
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AsyncExitStack
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pytest

from livekit.agents import AgentHandler, AgentServer, _core, voice
from livekit.agents.a2a import (  # type: ignore[attr-defined]  # a2a.py shadows its re-export
    AgentClient,
    TaskUpdate,
)
from livekit.agents.voice._session_bindings import ConsoleCall
from livekit.protocol.agent_pb.agent_session import (
    ASSISTANT,
    AgentSessionMessage,
    SessionRequest,
    SessionResponse,
)
from tests_new.e2e.resources import serving
from tests_new.fakes import FakeLLM, until

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples"
LIVEKIT_CREDENTIALS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


@pytest.fixture
def no_logged_tracebacks(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    yield
    _core.flush_logs()
    assert not [record for record in caplog.get_records("call") if record.exc_info]


@pytest.fixture
def no_livekit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LiveKit URL or keys in this process's environment, nor in the agents it starts."""
    for name in LIVEKIT_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)


class ConsolePeer:
    """The CLI's end of the ``lk agent console`` socket: keeps every message the agent sends,
    as the CLI does reports each flushed segment played, and sends the CLI's requests."""

    def __init__(self) -> None:
        self.addr = ""
        self.messages: list[AgentSessionMessage] = []
        self.connected = asyncio.Event()
        self._writer: asyncio.StreamWriter | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._tasks.add(asyncio.ensure_future(self._serve(reader, writer)))

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self.connected.set()
        try:
            while True:
                (size,) = struct.unpack(">I", await reader.readexactly(4))
                message = AgentSessionMessage.FromString(await reader.readexactly(size))
                self.messages.append(message)
                if message.WhichOneof("message") == "audio_playback_flush":
                    reply = AgentSessionMessage()
                    reply.audio_playback_finished.SetInParent()
                    body = reply.SerializeToString()
                    writer.write(struct.pack(">I", len(body)) + body)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def request(self, request: SessionRequest) -> SessionResponse:
        """Sends ``request`` as the CLI does and returns the agent's response to it."""
        await self.connected.wait()
        assert self._writer is not None
        body = AgentSessionMessage(request=request).SerializeToString()
        self._writer.write(struct.pack(">I", len(body)) + body)

        def response() -> SessionResponse | None:
            for m in self.messages:
                if m.WhichOneof("message") == "response":
                    if m.response.request_id == request.request_id:
                        return m.response
            return None

        await until(lambda: response() is not None, f"no response to {request.request_id}")
        answer = response()
        assert answer is not None
        return answer

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def kinds(self) -> list[str]:
        return [m.WhichOneof("message") or "" for m in self.messages]

    def assistant_texts(self) -> list[str]:
        texts = []
        for m in self.messages:
            if m.WhichOneof("message") != "event":
                continue
            if m.event.WhichOneof("event") != "conversation_item_added":
                continue
            item = m.event.conversation_item_added.item
            if item.WhichOneof("item") == "message" and item.message.role == ASSISTANT:
                texts.append("".join(c.text for c in item.message.content))
        return texts


@pytest.fixture
async def console(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ConsolePeer]:
    """The CLI's console socket: ``console.addr`` for ``--connect-addr``; a session started in
    this process claims it, as under ``lk agent console``."""
    peer = ConsolePeer()
    server = await asyncio.start_server(peer.accept, "127.0.0.1", 0)
    peer.addr = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    monkeypatch.setattr(ConsoleCall, "addr", peer.addr)
    try:
        yield peer
    finally:
        server.close()
        await peer.aclose()


@pytest.fixture
def agent_module(tmp_path: Path) -> Callable[[str], Path]:
    """Writes ``source`` as ``agent.py``; run it with ``python agent.py <command>``. It imports
    ``tests_new.fakes`` like any test does."""

    def write(source: str) -> Path:
        path = tmp_path / "agent.py"
        path.write_text(source)
        return path

    return write


@pytest.fixture
def agent_env() -> dict[str, str]:
    """The environment an agent module runs in: this process's, with the repository importable."""
    return {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}


@pytest.fixture
def source_module(monkeypatch: pytest.MonkeyPatch) -> Callable[[str, Path], ModuleType]:
    """Imports a file as ``name``; the module leaves ``sys.modules`` with the test."""

    def load(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def load_example(
    monkeypatch: pytest.MonkeyPatch, source_module: Callable[[str, Path], ModuleType]
) -> Callable[[str, FakeLLM], ModuleType]:
    """Loads an example's backend with every agent's model string resolving to ``model``."""

    def load(name: str, model: FakeLLM) -> ModuleType:
        directory = EXAMPLES / name
        monkeypatch.setenv("FRONTDESK_NOW", "2026-06-12T09:00")
        monkeypatch.setenv("HOTEL_TODAY", "2026-06-12")
        monkeypatch.delenv("CAL_API_KEY", raising=False)
        monkeypatch.setattr(voice, "llm_from_model_string", lambda _: model)
        dependency = {
            "frontdesk": "calendar_api",
            "hotel-receptionist": "hotel",
            "airline-support": "skyline",
        }[name]
        source_module(dependency, directory / f"{dependency}.py")
        return source_module(
            f"example_{name.replace('-', '_')}_backend", directory / "backend_agent.py"
        )

    return load


Ask = Callable[[str], Awaitable[TaskUpdate]]


class ChatWith(Protocol):
    async def __call__(self, endpoint: str, *handlers: AgentHandler) -> Ask: ...


@pytest.fixture
async def chat_with(
    no_livekit_credentials: None,
) -> AsyncIterator[ChatWith]:
    """``ask = await chat_with("backend", *handlers)`` serves ``handlers`` as ``lk agent console``
    does; ``await ask(text)`` sends one message to ``endpoint`` on one thread and returns the
    task's last update."""
    async with AsyncExitStack() as stack:

        async def chat(endpoint: str, *handlers: AgentHandler) -> Ask:
            server = AgentServer(agent_name="example")
            server.add(*handlers)
            url = await stack.enter_async_context(serving(server))
            client = AgentClient(f"{url}/{endpoint}")
            stack.push_async_callback(client.aclose)
            thread: str | None = None

            async def ask(text: str) -> TaskUpdate:
                nonlocal thread
                updates = [update async for update in await client.send(text, thread_id=thread)]
                thread = updates[-1].context_id
                return updates[-1]

            return ask

        yield chat
