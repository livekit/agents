"""`VoiceSession.history`: the native session's history, read as of the event being handled."""

import asyncio

import pytest

from livekit.agents import Agent, VoiceSession
from livekit.agents.llm import ChatContext, ChatItem
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_io import FakeAudioOutput
from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


def _session() -> VoiceSession:
    session = VoiceSession(
        vad=None, tts=FakeTTS(fake_audio_duration=0.1), turn_handling={"turn_detection": None}
    )
    session.output.audio = FakeAudioOutput()
    return session


def _texts(items: list[ChatItem]) -> list[str | None]:
    return [item.text_content for item in items if item.type == "message"]


async def test_a_callback_reads_the_history_as_of_its_event() -> None:
    session = _session()
    later, both = asyncio.Event(), asyncio.Event()
    seen: list[tuple[ConversationItemAddedEvent, list[ChatItem]]] = []
    read_later: list[asyncio.Task[list[ChatItem]]] = []

    async def read_after_the_next_item() -> list[ChatItem]:
        await later.wait()
        return session.history.items

    @session.on("conversation_item_added")
    def on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type != "message":
            return
        seen.append((ev, session.history.items))
        if len(seen) == 2:
            both.set()
        # a task the callback starts reads the same history, even once later items arrived
        read_later.append(asyncio.ensure_future(read_after_the_next_item()))

    history = ChatContext.empty()
    history.add_message(role="user", content="My booking reference is ABC123.")
    async with session:
        await session.start(Agent(instructions="Greet.", name="desk"), chat_ctx=history)
        await asyncio.wait_for(session.say("One moment."), timeout=5)
        await asyncio.wait_for(session.say("Anything else?"), timeout=5)
        await asyncio.wait_for(both.wait(), timeout=5)
        later.set()
        [first, second] = [items for _, items in seen]
        assert _texts(first) == ["My booking reference is ABC123.", "One moment."]
        assert _texts(second) == [*_texts(first), "Anything else?"]
        assert [_texts(await task) for task in read_later] == [_texts(first), _texts(second)]

        # outside a callback the history is the latest, over the same objects
        latest = session.history.items
        assert _texts(latest) == _texts(second)
        for ev, items in seen:
            assert ev.item is items[-1]
            assert any(item is ev.item for item in latest)
        assert all(a is b for a, b in zip(session.history.items, latest, strict=True))


async def test_editing_the_returned_history_does_not_change_the_session() -> None:
    session = _session()
    assert session.history.items == [], "empty before start()"
    async with session:
        await session.start(Agent(instructions="Greet.", name="desk"))
        await asyncio.wait_for(session.say("One moment."), timeout=5)
        session.history.add_message(role="user", content="not said")
        assert _texts(session.history.items) == ["One moment."]


async def test_a_pre_start_update_agent_history_is_the_start_history() -> None:
    session = _session()
    history = ChatContext.empty()
    history.add_message(role="user", content="carried over")
    agent = Agent(instructions="Greet.", name="desk")
    session.update_agent(agent, chat_ctx=history)
    async with session:
        await session.start(agent)
        assert _texts(session.history.items) == ["carried over"]
