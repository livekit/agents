from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    llm,
)
from livekit.agents.llm.chat_context import ChatMessage

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_session import FakeActions
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class SimpleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def _make_session() -> AgentSession:
    """Create a minimal session without TranscriptSynchronizer wrapping."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "placeholder")
    actions.add_llm("ok", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)

    user_speeches = actions.get_user_speeches(speed_factor=1.0)
    llm_responses = actions.get_llm_responses(speed_factor=1.0)
    tts_responses = actions.get_tts_responses(speed_factor=1.0)

    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.5,
            min_speech_duration=0.05,
        ),
        stt=FakeSTT(fake_user_speeches=user_speeches),
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        turn_handling={"turn_detection": None},
        aec_warmup_duration=None,
    )

    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()

    return session


async def _close(session: AgentSession) -> None:
    """Drain and close session, letting background tasks finish."""
    await asyncio.sleep(5)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()


async def test_add_conversation_item_appears_in_history() -> None:
    """A message added via add_conversation_item is retrievable from session.history."""
    session = _make_session()
    await session.start(SimpleAgent())

    msg = ChatMessage(role="user", content=["injected text"])
    result = await session.add_conversation_item(msg)

    assert result is True
    found = session.history.get_by_id(msg.id)
    assert found is not None
    assert found.text_content == "injected text"
    assert found.role == "user"

    await _close(session)


async def test_add_conversation_item_emits_event() -> None:
    """add_conversation_item fires the conversation_item_added event."""
    session = _make_session()

    received: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", received.append)

    await session.start(SimpleAgent())

    msg = ChatMessage(role="user", content=["event test"])
    await session.add_conversation_item(msg)

    matching = [e for e in received if e.item.id == msg.id]
    assert len(matching) == 1
    assert matching[0].item.role == "user"
    assert matching[0].item.text_content == "event test"

    await _close(session)


async def test_add_conversation_item_visible_to_agent() -> None:
    """A message added via add_conversation_item appears in the active agent's chat context."""
    session = _make_session()
    agent = SimpleAgent()
    await session.start(agent)

    msg = ChatMessage(role="user", content=["agent-visible"])
    await session.add_conversation_item(msg)

    found = agent.chat_ctx.get_by_id(msg.id)
    assert found is not None
    assert found.text_content == "agent-visible"

    await _close(session)


async def test_add_conversation_item_dedup_by_id() -> None:
    """Adding the same message ID twice is idempotent — only one copy in history."""
    session = _make_session()
    agent = SimpleAgent()
    await session.start(agent)

    received: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", received.append)

    msg = ChatMessage(role="user", content=["dedup test"])
    first = await session.add_conversation_item(msg)
    second = await session.add_conversation_item(msg)

    assert first is True
    assert second is False

    matches = [item for item in session.history.items if item.id == msg.id]
    assert len(matches) == 1
    assert [e.item.id for e in received] == [msg.id]
    assert agent.chat_ctx.get_by_id(msg.id) is not None

    await _close(session)


async def test_add_conversation_item_before_start() -> None:
    """add_conversation_item works on a session that hasn't been started yet.

    Pins the documented limitation: with no agent running, the item reaches
    session history and the event, but no agent context — items added before
    start are never backfilled into the agent's context once it starts.
    """
    session = _make_session()

    received: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", received.append)

    msg = ChatMessage(role="user", content=["pre-start"])
    result = await session.add_conversation_item(msg)

    assert result is True
    assert session.history.get_by_id(msg.id) is not None

    matching = [e for e in received if e.item.id == msg.id]
    assert len(matching) == 1

    # the item is never backfilled into the agent's context once it starts
    agent = SimpleAgent()
    await session.start(agent)
    assert agent.chat_ctx.get_by_id(msg.id) is None

    await _close(session)


async def test_add_conversation_item_realtime_dedup_by_id() -> None:
    """Realtime dedup checks session history: a repeated id is a full no-op."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        agent = SimpleAgent()
        await session.start(agent)
        rt_session = model.active_session
        assert rt_session is not None

        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)

        msg = ChatMessage(role="user", content=["rt dedup"])
        assert await session.add_conversation_item(msg) is True
        assert rt_session.chat_ctx.get_by_id(msg.id) is not None

        assert await session.add_conversation_item(msg) is False

        assert session.history.get_by_id(msg.id) is not None
        provider_item = rt_session.chat_ctx.get_by_id(msg.id)
        agent_item = agent.chat_ctx.get_by_id(msg.id)
        assert isinstance(provider_item, ChatMessage)
        assert isinstance(agent_item, ChatMessage)
        assert (provider_item.role, provider_item.text_content) == ("user", "rt dedup")
        assert (agent_item.role, agent_item.text_content) == ("user", "rt dedup")
        assert [e.item.id for e in received] == [msg.id]


async def test_add_conversation_item_realtime_pushes_to_rt_session() -> None:
    """On a realtime session the added item is pushed to the live provider context."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        agent = SimpleAgent()
        await session.start(agent)
        rt_session = model.active_session
        assert rt_session is not None

        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)

        msg = ChatMessage(role="user", content=["rt injected"])
        assert await session.add_conversation_item(msg) is True

        assert rt_session.chat_ctx.get_by_id(msg.id) is not None
        assert session.history.get_by_id(msg.id) is not None
        assert agent.chat_ctx.get_by_id(msg.id) is not None
        assert [e.item.id for e in received] == [msg.id]


async def test_add_conversation_item_realtime_non_mutable_raises() -> None:
    """Realtime models without mutable chat context refuse the add without side effects."""
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(audio_output=False, mutable_chat_context=False)
    )
    async with AgentSession(llm=model) as session:
        await session.start(SimpleAgent())
        rt_session = model.active_session
        assert rt_session is not None

        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)

        msg = ChatMessage(role="user", content=["non mutable"])
        with pytest.raises(llm.RealtimeError):
            await session.add_conversation_item(msg)

        assert session.history.get_by_id(msg.id) is None
        assert rt_session.chat_ctx.get_by_id(msg.id) is None
        assert received == []


async def test_add_conversation_item_realtime_push_failure_leaves_state_clean() -> None:
    """A failed realtime push mutates nothing, so retrying the same item id succeeds."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        await session.start(SimpleAgent())
        rt_session = model.active_session
        assert rt_session is not None

        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)

        rt_session.update_error = llm.RealtimeError("push timed out")
        msg = ChatMessage(role="user", content=["push failed"])
        with pytest.raises(llm.RealtimeError):
            await session.add_conversation_item(msg)

        assert session.history.get_by_id(msg.id) is None
        assert received == []

        rt_session.update_error = None
        assert await session.add_conversation_item(msg) is True
        assert session.history.get_by_id(msg.id) is not None
        assert rt_session.chat_ctx.get_by_id(msg.id) is not None
        assert [e.item.id for e in received] == [msg.id]


async def test_add_conversation_item_realtime_serializes_distinct_items() -> None:
    """Overlapping additions retain every distinct item in provider and local contexts."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        agent = SimpleAgent()
        await session.start(agent)
        rt_session = model.active_session

        release_update = asyncio.Event()
        rt_session.block_update_chat_ctx = release_update
        rt_session.update_chat_ctx_entered.clear()
        first = ChatMessage(role="user", content=["first concurrent item"])
        second = ChatMessage(role="user", content=["second concurrent item"])

        first_task = asyncio.create_task(session.add_conversation_item(first))
        await rt_session.update_chat_ctx_entered.wait()
        second_started = asyncio.Event()

        async def add_second() -> bool:
            second_started.set()
            return await session.add_conversation_item(second)

        second_task = asyncio.create_task(add_second())
        await second_started.wait()
        release_update.set()

        assert await asyncio.gather(first_task, second_task) == [True, True]
        for item in (first, second):
            assert rt_session.chat_ctx.get_by_id(item.id) is not None
            assert agent.chat_ctx.get_by_id(item.id) is not None
            assert session.history.get_by_id(item.id) is not None


async def test_add_conversation_item_realtime_serializes_same_id() -> None:
    """Overlapping same-ID additions commit and emit exactly once."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        agent = SimpleAgent()
        await session.start(agent)
        rt_session = model.active_session

        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)

        release_update = asyncio.Event()
        rt_session.block_update_chat_ctx = release_update
        rt_session.update_chat_ctx_entered.clear()
        item = ChatMessage(role="user", content=["same concurrent item"])

        first_task = asyncio.create_task(session.add_conversation_item(item))
        await rt_session.update_chat_ctx_entered.wait()
        second_started = asyncio.Event()

        async def add_same_item() -> bool:
            second_started.set()
            return await session.add_conversation_item(item)

        second_task = asyncio.create_task(add_same_item())
        await second_started.wait()
        release_update.set()

        assert sorted(await asyncio.gather(first_task, second_task)) == [False, True]
        assert sum(entry.id == item.id for entry in rt_session.chat_ctx.items) == 1
        assert sum(entry.id == item.id for entry in agent.chat_ctx.items) == 1
        assert sum(entry.id == item.id for entry in session.history.items) == 1
        assert [event.item.id for event in received] == [item.id]


async def test_add_conversation_item_uses_agent_realtime_model() -> None:
    """An agent-level realtime override receives the added item."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    agent = Agent(instructions="agent override", llm=model)
    async with AgentSession() as session:
        await session.start(agent)
        item = ChatMessage(role="user", content=["agent-level realtime"])

        assert await session.add_conversation_item(item) is True
        assert model.active_session.chat_ctx.get_by_id(item.id) is not None
        assert agent.chat_ctx.get_by_id(item.id) is not None
        assert session.history.get_by_id(item.id) is not None


async def test_add_conversation_item_uses_agent_realtime_capabilities() -> None:
    """An immutable agent-level realtime override rejects the item without side effects."""
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(audio_output=False, mutable_chat_context=False)
    )
    agent = Agent(instructions="agent override", llm=model)
    async with AgentSession() as session:
        await session.start(agent)
        item = ChatMessage(role="user", content=["agent-level immutable"])
        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)

        with pytest.raises(llm.RealtimeError):
            await session.add_conversation_item(item)

        assert model.active_session.chat_ctx.get_by_id(item.id) is None
        assert agent.chat_ctx.get_by_id(item.id) is None
        assert session.history.get_by_id(item.id) is None
        assert received == []


async def test_add_conversation_item_realtime_is_atomic_during_handoff() -> None:
    """A handoff cannot split an addition across the old and new agent contexts."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        await session.start(SimpleAgent())
        rt_session = model.active_session

        original_update_chat_ctx = rt_session.update_chat_ctx
        release_update = asyncio.Event()
        first_update_entered = asyncio.Event()
        handoff_update_completed = asyncio.Event()
        update_count = 0
        new_only: ChatMessage | None = None

        async def pause_first_update(chat_ctx: llm.ChatContext) -> None:
            nonlocal update_count
            update_count += 1
            if update_count == 1:
                first_update_entered.set()
                await release_update.wait()
            await original_update_chat_ctx(chat_ctx)
            if new_only is not None and chat_ctx.get_by_id(new_only.id) is not None:
                handoff_update_completed.set()

        rt_session.update_chat_ctx = pause_first_update
        added = ChatMessage(role="user", content=["added during handoff"])
        add_task = asyncio.create_task(session.add_conversation_item(added))
        await first_update_entered.wait()

        new_only = ChatMessage(role="user", content=["new agent context"])
        new_agent = Agent(
            instructions="replacement agent",
            chat_ctx=llm.ChatContext([new_only]),
        )
        session.update_agent(new_agent)
        for _ in range(10):
            await asyncio.sleep(0)
        release_update.set()

        assert await add_task is True
        await handoff_update_completed.wait()
        assert rt_session.chat_ctx.get_by_id(new_only.id) is not None
        assert new_agent.chat_ctx.get_by_id(new_only.id) is not None
        assert session.history.get_by_id(added.id) is not None


async def test_add_conversation_item_realtime_echo_before_failure_is_not_committed() -> None:
    """A provider echo before a failed push cannot mutate local state."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        agent = SimpleAgent()
        await session.start(agent)
        rt_session = model.active_session

        received: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", received.append)
        item = ChatMessage(role="user", content=["echo then fail"])

        async def echo_then_fail(chat_ctx: llm.ChatContext) -> None:
            pushed = chat_ctx.get_by_id(item.id)
            assert pushed is not None
            rt_session.emit(
                "remote_item_added",
                llm.RemoteItemAddedEvent(previous_item_id=None, item=pushed),
            )
            raise llm.RealtimeError("push failed after echo")

        rt_session.update_chat_ctx = echo_then_fail

        with pytest.raises(llm.RealtimeError):
            await session.add_conversation_item(item)

        assert rt_session.chat_ctx.get_by_id(item.id) is None
        assert agent.chat_ctx.get_by_id(item.id) is None
        assert session.history.get_by_id(item.id) is None
        assert received == []
