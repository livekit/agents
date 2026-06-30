from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from livekit.agents.llm import ChatContext
from livekit.plugins import google_adk
from livekit.plugins.google_adk import adk as google_adk_impl


class FakePart:
    def __init__(self, *, text: str | None = None, thought: bool = False) -> None:
        self.text = text
        self.thought = thought

    @classmethod
    def from_text(cls, *, text: str) -> FakePart:
        return cls(text=text)


class FakeContent:
    def __init__(self, *, role: str, parts: list[FakePart]) -> None:
        self.role = role
        self.parts = parts


class FakeEvent:
    def __init__(
        self,
        *,
        invocation_id: str = "",
        author: str = "",
        content: FakeContent | None = None,
        partial: bool = False,
        error_message: str | None = None,
    ) -> None:
        self.invocation_id = invocation_id
        self.author = author
        self.content = content
        self.partial = partial
        self.error_message = error_message


@dataclass
class FakeSession:
    app_name: str
    user_id: str
    id: str
    events: list[FakeEvent] = field(default_factory=list)


class FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str, str], FakeSession] = {}

    async def create_session(self, *, app_name: str, user_id: str, session_id: str) -> FakeSession:
        session = FakeSession(app_name=app_name, user_id=user_id, id=session_id)
        self.sessions[(app_name, user_id, session_id)] = session
        return session

    async def append_event(self, session: FakeSession, event: FakeEvent) -> FakeEvent:
        if not event.partial:
            session.events.append(event)
        return event

    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        self.sessions.pop((app_name, user_id, session_id), None)


class FakeRunner:
    def __init__(self, *, agent: Any = None, app: Any = None, app_name: str = "FakeApp") -> None:
        self.agent = agent if agent is not None else app.root_agent
        self.app_name = app_name
        self.session_service = FakeSessionService()
        self.closed = False

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: FakeContent,
        run_config: Any | None = None,
    ):
        session = self.session_service.sessions[(self.app_name, user_id, session_id)]
        await self.session_service.append_event(
            session,
            FakeEvent(
                invocation_id="current",
                author="user",
                content=new_message,
            ),
        )

        async for event in self.agent.run(session=session, new_message=new_message, run_config=run_config):
            yield event

    async def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class FakeADKModules:
    InMemoryRunner: type[Any] = FakeRunner
    Content: type[Any] = FakeContent
    Event: type[Any] = FakeEvent
    Part: type[Any] = FakePart


class StreamingEchoAgent:
    name = "echo-agent"
    model = "fake-model"

    async def run(self, *, session: FakeSession, new_message: FakeContent, run_config: Any | None = None):
        del session, run_config
        prompt = new_message.parts[0].text or ""
        midpoint = len(prompt) // 2
        yield FakeEvent(
            invocation_id="1",
            author=self.name,
            content=FakeContent(role="model", parts=[FakePart(text=prompt[:midpoint])]),
            partial=True,
        )
        yield FakeEvent(
            invocation_id="1",
            author=self.name,
            content=FakeContent(role="model", parts=[FakePart(text=prompt)]),
        )


class HistoryAgent:
    name = "history-agent"
    model = "fake-model"

    async def run(self, *, session: FakeSession, new_message: FakeContent, run_config: Any | None = None):
        del new_message, run_config
        history = []
        for event in session.events:
            if not event.content:
                continue
            texts = [part.text for part in event.content.parts if part.text]
            if texts:
                history.append(f"{event.content.role}:{''.join(texts)}")

        yield FakeEvent(
            invocation_id="2",
            author=self.name,
            content=FakeContent(role="model", parts=[FakePart(text=" | ".join(history))]),
        )


async def collect_chunks(stream) -> str:
    chunks: list[str] = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    return "".join(chunks)


@pytest.fixture(autouse=True)
def fake_adk_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    google_adk_impl._adk_modules.cache_clear()
    monkeypatch.setattr(google_adk_impl, "_adk_modules", lambda: FakeADKModules())


@pytest.mark.asyncio
async def test_google_adk_adapter_streams_text_and_injects_instructions() -> None:
    adapter = google_adk.LLMAdapter(agent=StreamingEchoAgent())

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Be concise.")
    chat_ctx.add_message(role="developer", content="Answer in one sentence.")
    chat_ctx.add_message(role="user", content="Hello from LiveKit")

    text = await collect_chunks(adapter.chat(chat_ctx=chat_ctx))

    assert "System instructions:\nBe concise." in text
    assert "Developer instructions:\nAnswer in one sentence." in text
    assert text.endswith("User message:\nHello from LiveKit")


@pytest.mark.asyncio
async def test_google_adk_adapter_rebuilds_history_from_chat_context() -> None:
    adapter = google_adk.LLMAdapter(agent=HistoryAgent())

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    chat_ctx.add_message(role="assistant", content="Hello there")
    chat_ctx.add_message(role="user", content="How are you?")

    text = await collect_chunks(adapter.chat(chat_ctx=chat_ctx))

    assert "user:Hi" in text
    assert "model:Hello there" in text
    assert "user:How are you?" in text


@pytest.mark.asyncio
async def test_google_adk_adapter_closes_owned_runner() -> None:
    adapter = google_adk.LLMAdapter(agent=StreamingEchoAgent())

    await adapter.aclose()

    assert adapter._runner.closed is True
