import asyncio

import pytest

from livekit.agents import Agent, VoiceSession
from livekit.agents.llm import ChatContext

pytestmark = pytest.mark.unit


async def test_agent_switch_preserves_the_conversation_history() -> None:
    history = ChatContext.empty()
    history.add_message(role="user", content="My booking reference is ABC123.")
    first = Agent(instructions="Welcome the caller.", name="reception")
    second = Agent(instructions="Help with bookings.", name="bookings")
    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(first, chat_ctx=history)
        session.update_agent(second)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        messages = [item.text_content for item in session.history.items if item.type == "message"]
        assert "My booking reference is ABC123." in messages
        assert session.current_agent is second


async def test_agent_switch_preserves_session_listening_options() -> None:
    async with VoiceSession(
        vad=None,
        turn_detection="manual",
        endpointing_opts={"min_delay": 0.3, "max_delay": 2.0},
    ) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        session.update_agent(Agent(instructions="Help with bookings.", name="bookings"))
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        assert session.options["turn_detection_mode"] == "manual"
        assert session.options["endpointing"]["min_delay"] == 0.3


async def test_temporary_options_restore_the_base_after_an_exception() -> None:
    session = VoiceSession(vad=None, turn_handling={"turn_detection": None})
    try:
        with pytest.raises(RuntimeError, match="interaction failed"):
            async with session.override_options(turn_detection="manual"):
                session.update_options(turn_detection="vad")
                raise RuntimeError("interaction failed")
        assert session.options["turn_detection_mode"] == "vad"
    finally:
        await session.aclose()


async def test_close_waits_for_recording_before_a_late_close_event() -> None:
    from typing import cast
    from unittest.mock import AsyncMock

    from livekit.agents import _core
    from livekit.agents.voice.events import CloseReason

    session = VoiceSession(vad=None, turn_handling={"turn_detection": None})
    core = AsyncMock()
    session._core = cast(_core.VoiceSession, core)
    finishing = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Recorder:
        async def finish(self) -> None:
            nonlocal calls
            calls += 1
            finishing.set()
            await release.wait()

    session._recorder = cast(_core.SessionRecorder, Recorder())
    closing = asyncio.create_task(session.aclose())
    try:
        await asyncio.sleep(0)
        assert not closing.done(), "close must wait for recording even before its close event"
        await asyncio.wait_for(finishing.wait(), timeout=1)
        session._on_close({"reason": CloseReason.USER_INITIATED.value})
        release.set()
        await closing
        assert calls == 1, "a late close event must join the already-owned finalization"
    finally:
        release.set()
        await closing


async def test_recording_failure_does_not_fail_session_close() -> None:
    from typing import cast
    from unittest.mock import AsyncMock

    from livekit.agents import _core

    session = VoiceSession(vad=None, turn_handling={"turn_detection": None})
    session._core = cast(_core.VoiceSession, AsyncMock())
    failure = RuntimeError("recording capacity exceeded: staging bytes")

    class Recorder:
        async def finish(self) -> None:
            raise failure

    session._recorder = cast(_core.SessionRecorder, Recorder())
    await session.aclose()
    assert session._recording_finished is not None
    assert session._recording_finished.exception() is failure, "the job still reports this failure"
