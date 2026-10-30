"""Voice turns share the durable lane and explicitly saved fields with text workers."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from livekit.agents import Agent, RunContext, StopResponse, VoiceSession, persisted
from livekit.agents._generated.storage_class import ItemRow
from livekit.agents.llm import ChatContext, ChatMessage, FunctionToolCall, LLMStream
from livekit.agents.storage import SQLiteStorage, Storage
from livekit.agents.storage._protocol import item_to_payload
from livekit.agents.storage.context_store import ContextStore
from livekit.agents.storage.resolve import Thread

from .fake_llm import FakeLLM, FakeLLMResponse
from .storage_fixture import new_session

pytestmark = pytest.mark.unit


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
        await run.commit(
            [ItemRow(id=item.id, message_id=None, payload=item_to_payload(item)) for item in items]
        )
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
    thread = Thread(voice_worker, thread_id, agent_name="desk")
    opening = ChatContext(
        [
            ChatMessage(
                id="override", role="assistant", content=["application opening"], created_at=1
            )
        ]
    )
    async with VoiceSession(llm=model, thread=thread) as session:
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
async def test_voice_explicit_save_joins_reply_commit_and_releases_the_lane(
    thread_storage: Storage,
    no_reply: bool,
) -> None:
    thread_id = await new_session(thread_storage, "desk")
    thread = Thread(thread_storage, thread_id, agent_name="desk")
    app = Application()
    agent = Agent[Application](instructions="desk", name="desk")

    @agent.tool
    async def save(ctx: RunContext[Application]) -> str:
        """Save the caller's draft."""
        assert ctx.thread is thread and ctx.context is app
        ctx.context.draft.count += 1
        await ctx.store.save_context(ctx.context)
        assert not (await thread.chat_history()).items, "save is staged with the voice turn"
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
    async with VoiceSession(llm=model, context=app, thread=thread) as session:
        await session.start(agent)
        await session.store.load_context(app)
        await session._run(user_input="save")
        core = await thread_storage._open(thread_id)
        try:
            lane = await core.lane_info("desk")
            assert lane is not None and lane.lease_until == 0
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


async def test_storage_failure_at_voice_start_is_logged_without_ending_the_call(
    thread_storage: Storage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    thread_id = await new_session(thread_storage, "desk")
    thread = Thread(thread_storage, thread_id, agent_name="desk")
    await thread_storage.threads.delete_thread(thread_id)
    model = ObservedLLM()
    async with VoiceSession(llm=model, thread=thread) as session:
        await session.start(Agent(instructions="desk", name="desk"))
        result = await session._run(user_input="hello")
        assert result.output == "hi there"
        assert "the voice lane could not be loaded" in caplog.text


@pytest.mark.parametrize("deleted_thread", [False, True])
async def test_call_without_a_loaded_lane_keeps_its_context_in_memory(
    thread_storage: Storage,
    deleted_thread: bool,
) -> None:
    thread = None
    if deleted_thread:
        thread_id = await new_session(thread_storage, "desk")
        thread = Thread(thread_storage, thread_id, agent_name="desk")
        await thread_storage.threads.delete_thread(thread_id)
    app = Application()
    async with VoiceSession(llm=ObservedLLM(), context=app, thread=thread) as session:
        await session.start(Agent(instructions="desk", name="desk"))
        await session.store.load_context(app)
        app.draft.count = 2
        await session.store.save_context(app)
        restored = Application()
        await session.store.load_context(restored)
        assert restored.draft.count == 2


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
    thread = Thread(thread_storage, thread_id, agent_name="desk")
    app = Application()
    agent = Agent[Application](instructions="desk", name="desk")

    @agent.tool
    async def save(ctx: RunContext[Application]) -> str:
        """Stage a draft before the next provider request fails."""
        ctx.context.draft.count = 1
        await ctx.store.save_context(ctx.context)
        return "saved"

    async with VoiceSession(llm=FailingAfterTool(), context=app, thread=thread) as session:
        await session.start(agent)
        await session.store.load_context(app)
        await session.store.save_context(app)
        speech = session.generate_reply(user_input="save")
        done = asyncio.Event()
        speech.add_done_callback(lambda _: done.set())
        await asyncio.wait_for(done.wait(), timeout=5)
        assert speech.exception() is not None
        assert not (await thread.chat_history()).items
        await session.store.load_context(app)
        assert app.draft.count == 0
        result = await session._run(user_input="hello")
        assert result.output == "recovered"
    assert texts(await thread.chat_history()) == ["hello", "recovered"]


async def test_a_lost_voice_lease_while_the_host_interrupts_completes_the_call(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The lane's lease is lost while the host keeps calling ``interrupt()`` from another thread:
    the core logs the loss while it steps the host's requests, and the call still completes
    (a log forwarded to the host under that step's lock once deadlocked the session)."""
    storage = SQLiteStorage(
        directory=tmp_path,
        lease_ttl=timedelta(seconds=2),
        renew_interval=timedelta(milliseconds=100),
    )
    thread_id = await new_session(storage, "desk")
    thread = Thread(storage, thread_id, agent_name="desk")
    agent = Agent(instructions="desk", name="desk")

    @agent.tool
    async def hold(ctx: RunContext) -> str:
        """Hold the line."""
        handle = ctx.speech_handle
        # another holder takes the lane while this turn holds it
        with sqlite3.connect(tmp_path / "threads" / f"{thread_id}.sqlite") as db:
            db.execute("UPDATE lanes SET lease_token = 'stolen', lease_holder = 'other'")
        stop = threading.Event()

        def interrupt_until_stopped() -> None:
            while not stop.is_set():
                handle.interrupt(force=True)

        host = threading.Thread(target=interrupt_until_stopped)
        host.start()
        try:
            async with asyncio.timeout(10):
                while "the voice lane lease was lost" not in caplog.text:
                    await asyncio.sleep(0.01)
        finally:
            stop.set()
            await asyncio.to_thread(host.join)
        return "held"

    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hold",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="hold", arguments="{}", call_id="hold-call")],
            ),
            FakeLLMResponse(input="held", content="still here", ttft=0, duration=0),
            FakeLLMResponse(input="again", content="hello again", ttft=0, duration=0),
        ]
    )
    async with VoiceSession(llm=model, thread=thread) as session:
        await session.start(agent)
        await asyncio.wait_for(session._run(user_input="hold"), 20)
        assert "the voice lane lease was lost" in caplog.text
        result = await asyncio.wait_for(session._run(user_input="again"), 20)
        assert result.output == "hello again"
