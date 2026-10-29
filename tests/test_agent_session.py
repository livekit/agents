import asyncio

import pytest

from livekit.agents import Agent, VoiceSession, utils
from livekit.agents.llm import ChatContext

from .fake_io import FakeAudioOutput
from .fake_tts import FakeTTS

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


async def test_live_session_changes_interruption_and_delivery_settings() -> None:
    async def interruption(**expected: object) -> None:
        # the running core applies a change on its own loop
        for _ in range(100):
            if expected.items() <= session.options["interruption"].items():
                return
            await asyncio.sleep(0.01)
        assert expected.items() <= session.options["interruption"].items()

    async with VoiceSession(vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        session.update_options(
            interruption={"min_duration": 1.5},
            min_consecutive_speech_delay=0.4,
            tts_text_transforms=None,
        )
        async with session.override_options(interruption={"enabled": False}):
            async with session.override_options(interruption={"min_words": 2}):
                await interruption(enabled=False, min_words=2, min_duration=1.5)
            await interruption(enabled=False, min_words=0)
        await interruption(enabled=True, min_words=0, min_duration=1.5)
        assert session.options["min_consecutive_speech_delay"] == 0.4
        assert session.options["tts_text_transforms"] is None


def _voiced_session(tts: FakeTTS) -> VoiceSession:
    session = VoiceSession(vad=None, tts=tts, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    return session


def _synthesized(tts: FakeTTS) -> int:
    streams = 0
    while True:
        try:
            tts.stream_ch.recv_nowait()
        except utils.aio.ChanEmpty:
            return streams
        streams += 1


async def test_update_options_replaces_the_session_voice_in_place() -> None:
    first, second = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(first) as session:
        agent = Agent(instructions="Welcome the caller.", name="reception")
        await session.start(agent)
        session.update_options(tts=second)
        await asyncio.wait_for(session.say("One moment."), timeout=5)
        assert session.tts is second
        assert session.current_agent is agent
        assert _synthesized(first) == 0, "the replaced voice must not speak"
        assert _synthesized(second) == 1


async def test_override_options_speaks_with_the_layer_then_the_base() -> None:
    base, specialist = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(base) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        async with session.override_options(tts=specialist):
            await asyncio.wait_for(session.say("Let me read the terms back."), timeout=5)
        await asyncio.wait_for(session.say("Anything else?"), timeout=5)
        assert session.tts is base
        assert _synthesized(specialist) == 1
        assert _synthesized(base) == 1


async def test_close_waits_for_recording_before_a_late_close_event() -> None:
    from typing import cast
    from unittest.mock import AsyncMock

    from livekit.agents import _core
    from livekit.agents._generated.session_wire import CloseWire
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
        session._on_close(CloseWire(reason=CloseReason.USER_INITIATED, error=None))
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
