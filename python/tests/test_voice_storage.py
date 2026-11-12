"""Voice turns restore durable history and explicitly saved application fields."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from livekit.agents import Agent, RunContext, SessionState, StopResponse, VoiceSession, persisted
from livekit.agents._generated.storage_class import ItemRow
from livekit.agents.llm import ChatContext, ChatMessage, FunctionToolCall, LLMStream
from livekit.agents.storage import Storage, Thread, ThreadNotFound
from livekit.agents.storage.session import ContextStore

from .fake_llm import FakeLLM, FakeLLMResponse
from .storage_fixture import new_session

pytestmark = pytest.mark.unit


async def bound(thread: Thread, app: Any = None) -> Any:
    activation = await thread._storage.threads._core_context_store(thread.id, thread._partition_key)
    store = ContextStore(
        activation=activation, thread=thread, scope_id=f"root:{thread._partition_key}"
    )
    return store.bind(app)


class ObservedLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(
            fake_responses=[
                FakeLLMResponse(input="hello", content="hi there", ttft=0, duration=0),
                FakeLLMResponse(input="again", content="read the text", ttft=0, duration=0),
            ]
        )
        self.requests: list[ChatContext] = []

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
        self.requests.append(chat_ctx.copy())
        return super().chat(chat_ctx=chat_ctx, **kwargs)


async def replace_history(storage: Storage, thread_id: str, items: list[ChatMessage]) -> None:
    core = await storage._open(thread_id)
    try:
        run = await core.open_run("desk", f"edit-{items[-1].id}", channel="sms")
        await run.commit([ItemRow(id=item.id, message_id=None, payload=item) for item in items])
    finally:
        core.close()


def texts(context: ChatContext) -> list[str]:
    return [
        item.text_content or ""
        for item in context.items
        if item.type == "message" and item.role not in ("system", "developer")
    ]


@pytest.mark.parametrize("explicit_history", [False, True])
async def test_voice_replaces_changed_transport_before_the_model_reads(
    thread_storages: tuple[Storage, Storage],
    explicit_history: bool,
) -> None:
    voice_worker, text_worker = thread_storages
    thread_id = await new_session(voice_worker, "desk")
    await replace_history(
        text_worker,
        thread_id,
        [
            ChatMessage(id="old", role="assistant", content=["stored opening"], created_at=1),
        ],
    )
    model = ObservedLLM()
    agent = Agent(instructions="desk", name="desk")
    thread = Thread(voice_worker, thread_id, partition_key="desk")
    opening = ChatContext(
        [
            ChatMessage(
                id="override", role="assistant", content=["application opening"], created_at=1
            )
        ]
    )
    async with VoiceSession(llm=model, context=await bound(thread)) as session:
        await session.start(agent, chat_ctx=opening if explicit_history else None)
        assert texts(session.history) == [
            "application opening" if explicit_history else "stored opening"
        ]
        await session._run(user_input="hello")
        persisted_history = await thread.chat_history()
        assert texts(persisted_history)[-2:] == ["hello", "hi there"]
        hello = next(
            item
            for item in persisted_history.items
            if item.type == "message" and item.text_content == "hello"
        )
        # One text worker edits an id, deletes old rows and compacts them into a summary.
        await replace_history(
            text_worker,
            thread_id,
            [
                ChatMessage(id="summary", role="assistant", content=["compacted"], created_at=1),
                ChatMessage(id=hello.id, role="user", content=["edited hello"], created_at=2),
                ChatMessage(id="text", role="user", content=["a text"], created_at=3),
                ChatMessage(
                    id="text-answer", role="assistant", content=["texted back"], created_at=4
                ),
            ],
        )
        expected = texts(await thread.chat_history()) + ["again"]
        assert "edited hello" in expected and "compacted" in expected
        assert "hello" not in expected and "hi there" not in expected
        await session._run(user_input="again")
        assert texts(model.requests[-1]) == expected
        assert texts(session.history) == expected + ["read the text"]
        assert texts(session._agent_chat_ctx()) == expected + ["read the text"]
    assert texts(await thread.chat_history()) == expected + ["read the text"]


@dataclass
class Draft:
    count: int = 0


@dataclass
class Application:
    draft: Draft = persisted(default_factory=Draft)


@pytest.mark.parametrize("no_reply", [False, True])
async def test_voice_explicit_save_joins_reply_commit_and_releases_the_partition(
    thread_storage: Storage,
    no_reply: bool,
) -> None:
    thread_id = await new_session(thread_storage, "desk")
    thread = Thread(thread_storage, thread_id, partition_key="desk")
    app = Application()
    agent = Agent[Application](instructions="desk", name="desk")

    @agent.tool
    async def save(ctx: RunContext[Application]) -> str:
        """Save the caller's draft."""
        assert ctx.thread is thread and ctx.context is app
        ctx.context.draft.count += 1
        await ctx.store.save_context(ctx.context)
        assert "save" in texts(await thread.chat_history()), (
            "conversation admission is checkpointed"
        )
        activation = await thread_storage.threads._core_context_store(thread_id, "desk")
        snapshot_store = ContextStore(activation=activation, scope_id="root:desk")
        fresh = await snapshot_store.load_context(Application())
        assert fresh.draft.count == 0, "application save remains staged until turn settlement"
        snapshot_store._close()
        if no_reply:
            raise StopResponse()
        return "saved"

    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="save",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="save", arguments="{}", call_id="save-call")],
            ),
            FakeLLMResponse(input="saved", content="done", ttft=0, duration=0),
        ]
    )
    async with VoiceSession(llm=model, context=await bound(thread, app)) as session:
        await session.start(agent)
        await session.store.load_context(app)
        await session._run(user_input="save")
        core = await thread_storage._open(thread_id)
        try:
            partition_key = await core.partition_info("desk")
            assert partition_key is not None and partition_key.lease_expires_at_ms == 0
        finally:
            core.close()
        activation = await thread_storage.threads._core_context_store(thread_id, "desk")
        restored = Application()
        await ContextStore(activation=activation, scope_id="root:desk").load_context(restored)
        assert restored.draft.count == 1
        history = await thread.chat_history()
        assert any(item.type == "function_call_output" for item in history.items)
        assert texts(history)[0] == "save"
        assert texts(history)[-1] == ("save" if no_reply else "done")


# Catches: a call given a thread whose partition does not load continuing on a new local thread (a
# silent fork, a local file on an AgentDB server) instead of failing its start with the storage's
# own error, logged once with the job and the thread.
async def test_a_call_on_a_thread_that_does_not_load_fails_to_start(
    thread_storage: Storage,
) -> None:
    thread_id = await new_session(thread_storage, "desk")
    thread = Thread(thread_storage, thread_id, partition_key="desk")
    await thread_storage.threads.delete_thread(thread_id)
    with pytest.raises(ThreadNotFound):
        await bound(thread)


# Catches: a call without a thread writing a thread anywhere, a local SQLite file included,
# instead of keeping its context and history in the session's memory.
async def test_call_without_a_thread_runs_in_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = tmp_path / "local-threads"
    monkeypatch.setenv("LIVEKIT_AGENT_THREADS_DIR", str(local))
    app = Application()
    async with VoiceSession(llm=ObservedLLM(), context=app) as session:
        await session.start(Agent(instructions="desk", name="desk"))
        assert session.thread is None
        await session.store.load_context(app)
        app.draft.count = 2
        await session.store.save_context(app)
        restored = Application()
        await session.store.load_context(restored)
        assert restored.draft.count == 2
    assert not local.exists()


async def test_failed_voice_generation_abandons_saved_fields_before_the_next_turn(
    thread_storage: Storage,
) -> None:
    from livekit.agents import APIStatusError

    class FailingAfterTool(FakeLLM):
        def __init__(self) -> None:
            super().__init__(
                fake_responses=[
                    FakeLLMResponse(
                        input="save",
                        content="",
                        ttft=0,
                        duration=0,
                        tool_calls=[
                            FunctionToolCall(name="save", arguments="{}", call_id="failed-save")
                        ],
                    ),
                    FakeLLMResponse(input="hello", content="recovered", ttft=0, duration=0),
                ]
            )
            self.failed = False

        def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
            if (
                any(item.type == "function_call_output" for item in chat_ctx.items)
                and not self.failed
            ):
                self.failed = True
                raise APIStatusError("provider refused", status_code=401, retryable=False)
            return super().chat(chat_ctx=chat_ctx, **kwargs)

    thread_id = await new_session(thread_storage, "desk")
    thread = Thread(thread_storage, thread_id, partition_key="desk")
    app = Application()
    agent = Agent[Application](instructions="desk", name="desk")

    @agent.tool
    async def save(ctx: RunContext[Application]) -> str:
        """Stage a draft before the next provider request fails."""
        ctx.context.draft.count = 1
        await ctx.store.save_context(ctx.context)
        return "saved"

    async with VoiceSession(llm=FailingAfterTool(), context=await bound(thread, app)) as session:
        await session.start(agent)
        await session.store.load_context(app)
        await session.store.save_context(app)
        speech = session.generate_reply(user_input="save")
        done = asyncio.Event()
        speech.add_done_callback(lambda _: done.set())
        await asyncio.wait_for(done.wait(), timeout=5)
        assert speech.exception() is not None
        checkpointed = await thread.chat_history()
        assert texts(checkpointed) == ["save"]
        assert any(item.type == "function_call_output" for item in checkpointed.items)
        await session.store.load_context(app)
        assert app.draft.count == 0
        result = await session._run(user_input="hello")
        assert result.output == "recovered"
    assert texts(await thread.chat_history()) == ["save", "hello", "recovered"]


async def test_voice_binding_uses_the_preloaded_context_and_updates_session_in_memory() -> None:
    @dataclass
    class App:
        session: SessionState = persisted(default_factory=SessionState)

    app = App()
    store = ContextStore()
    await store.load_context(app)
    original = app.session
    before = await store.load_session()
    agent = Agent(name="desk", instructions="Help")
    async with VoiceSession(llm=ObservedLLM(), context=store.bind(app)) as session:
        await session.start(agent)
        assert session.context is app and session.store is store
        await session._run(user_input="hello")
        assert app.session is original and app.session.active_agent == "desk"
        assert before.kind == "empty"
        assert (await store.load_session()).kind == "voice"
    assert store._session_binding is None
