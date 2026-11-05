from __future__ import annotations

import asyncio

import pytest

from livekit.agents.voice.io import SpeechHandle

pytestmark = pytest.mark.unit


class _Core:
    """The core's handle as the host reads it: done as soon as the core published the speech's
    end, which is before the session's ``speech_done`` reaches the host loop."""

    host = None
    id = "speech_1"
    is_done = False

    def done(self) -> bool:
        return self.is_done


async def test_done_callbacks_run_on_speech_done_in_registration_order() -> None:
    core = _Core()
    handle = SpeechHandle(core)  # type: ignore[arg-type]
    calls: list[str] = []
    handle.add_done_callback(lambda _: calls.append("first"))
    core.is_done = True
    # added after the core ended the speech but before the session reported it done
    handle.add_done_callback(lambda _: calls.append("second"))
    await asyncio.sleep(0)
    assert calls == [], "nothing runs before the session reports the speech done"
    handle._mark_done()
    await asyncio.sleep(0)
    assert calls == ["first", "second"]


async def test_a_callback_added_after_the_core_ended_waits_for_speech_done() -> None:
    core = _Core()
    handle = SpeechHandle(core)  # type: ignore[arg-type]
    calls: list[str] = []
    core.is_done = True
    handle.add_done_callback(lambda _: calls.append("late"))
    await asyncio.sleep(0)
    assert calls == [], "ran before the session reported the speech done"
    handle._mark_done()
    assert calls == ["late"]


async def test_a_say_the_core_finished_before_returning_still_honours_a_removal() -> None:
    # `say()` releases the GIL: with no audio output the core can end the speech before `say()`
    # wraps it, while its `speech_done` still waits for the host loop.
    core = _Core()
    core.is_done = True
    handle = SpeechHandle(core)  # type: ignore[arg-type]
    calls: list[str] = []
    handle.add_done_callback(dropped := lambda _: calls.append("dropped"))
    handle.add_done_callback(lambda _: calls.append("kept"))
    handle.remove_done_callback(dropped)
    await asyncio.sleep(0)
    assert calls == [], "ran before the session reported the speech done"
    handle._mark_done()
    assert calls == ["kept"]


async def test_a_speech_cut_by_the_session_close_runs_its_done_callbacks() -> None:
    from livekit.agents import Agent, VoiceSession

    from .fake_llm import FakeLLM, FakeLLMResponse

    llm = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="hi there", ttft=30, duration=30)]
    )
    calls: list[bool] = []
    async with VoiceSession(llm=llm) as session:
        await session.start(Agent(instructions="desk", name="desk"))
        speech = session.generate_reply(user_input="hello")
        speech.add_done_callback(lambda handle: calls.append(handle.done()))
        await asyncio.sleep(0.1)
        assert calls == []
    await asyncio.sleep(0)
    assert calls == [True], "the close ended the speech without telling the host"
