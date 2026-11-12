from __future__ import annotations

import asyncio

import pytest

from livekit.agents.voice.io import SpeechHandle

pytestmark = pytest.mark.unit


class _Core:
    """The core's handle as the host reads it: done as soon as the core published the speech's
    end, which is before the session hands the host that end (``wait_reported``)."""

    host = None
    id = "speech_1"
    is_done = False

    def __init__(self) -> None:
        self.report: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def done(self) -> bool:
        return self.is_done

    def wait_reported(self) -> asyncio.Future[None]:
        return self.report


async def test_done_callbacks_run_once_the_end_is_reported_in_registration_order() -> None:
    core = _Core()
    handle = SpeechHandle(core)  # type: ignore[arg-type]
    calls: list[str] = []
    handle.add_done_callback(lambda _: calls.append("first"))
    core.is_done = True
    # added after the core ended the speech but before the session reported it done
    handle.add_done_callback(lambda _: calls.append("second"))
    await asyncio.sleep(0)
    assert calls == [], "nothing runs before the session reports the speech done"
    core.report.set_result(None)
    await asyncio.sleep(0)
    assert calls == ["first", "second"]
    handle.add_done_callback(lambda _: calls.append("late"))
    assert calls == ["first", "second"], "a callback added after the report runs soon, not now"
    await asyncio.sleep(0)
    assert calls == ["first", "second", "late"]


async def test_a_removed_callback_does_not_run_and_a_raising_one_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    core = _Core()
    handle = SpeechHandle(core)  # type: ignore[arg-type]
    calls: list[str] = []

    def fail(_: SpeechHandle) -> None:
        raise RuntimeError("callback failed")

    handle.add_done_callback(dropped := lambda _: calls.append("dropped"))
    handle.add_done_callback(fail)
    handle.add_done_callback(lambda _: calls.append("kept"))
    handle.remove_done_callback(dropped)
    core.report.set_result(None)
    await asyncio.sleep(0)
    assert calls == ["kept"]
    assert any(r.getMessage().startswith("error in done_callback") for r in caplog.records)


async def test_a_done_callback_runs_after_the_events_before_the_speech_end() -> None:
    from livekit.agents import Agent, VoiceSession

    from .fake_io import FakeAudioOutput
    from .fake_tts import FakeTTS

    session = VoiceSession(
        vad=None, tts=FakeTTS(fake_audio_duration=0.1), turn_handling={"turn_detection": None}
    )
    session.output.audio = FakeAudioOutput()
    added: list[str | None] = []
    at_done: list[list[str | None]] = []
    session.on(
        "conversation_item_added",
        lambda ev: ev.item.type == "message" and added.append(ev.item.text_content),
    )
    async with session:
        await session.start(Agent(instructions="desk", name="desk"))
        speech = session.say("One moment.")
        speech.add_done_callback(lambda _: at_done.append(list(added)))
        await speech
        async with asyncio.timeout(5):
            while not at_done:
                await asyncio.sleep(0.01)
    assert at_done == [["One moment."]], "the speech's message event was handled first"


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
