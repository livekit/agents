"""EndCallTool must not wait for an auto tool reply when the model already spoke.

GPT-Live (auto_tool_reply_generation=True) speaks the goodbye in the same turn that
calls end_call (#8010). The tool output ("say goodbye to the user") then asks the model
for a tool reply that never arrives as a new generation, so the old code sat through
the 5s auto-reply timeout: the caller heard silence, a repeated goodbye, or a second
end_call 4.9-6.3s after the goodbye ended.

Fix: when the turn that called end_call already carried assistant speech, shut down as
soon as that speech finishes. A silent end_call keeps the reply-wait path, because the
tool reply is the goodbye there.
"""

from __future__ import annotations

import asyncio
import unittest.mock
from collections.abc import Sequence

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, utils
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit]

_SAMPLE_RATE = 24000


def _generation(
    *,
    response_id: str,
    text: str,
    audio_duration: float,
    function_calls: Sequence[FunctionCall] = (),
) -> GenerationCreatedEvent:
    """A single-message generation carrying `audio_duration` seconds of silence."""
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])

    message_ch.send_nowait(
        MessageGeneration(
            message_id=f"{response_id}-message",
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    text_ch.send_nowait(text)
    text_ch.close()
    samples = int(_SAMPLE_RATE * audio_duration)
    audio_ch.send_nowait(
        rtc.AudioFrame(
            data=bytes(2 * samples),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=samples,
        )
    )
    audio_ch.close()
    for fnc_call in function_calls:
        function_ch.send_nowait(fnc_call)
    function_ch.close()

    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=True,
        response_id=response_id,
    )


async def test_end_call_same_turn_speech_shuts_down_without_reply_wait() -> None:
    """The #8010 shape: goodbye + end_call in one turn closes at playout end, not +5s."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    ect = EndCallTool(end_instructions=None)

    class GoodbyeAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test", tools=[ect])

    delayed_wait_started = asyncio.Event()

    _orig_delayed = ect._delayed_session_shutdown

    async def _spy_delayed_shutdown(ctx: object) -> None:
        delayed_wait_started.set()
        await _orig_delayed(ctx)  # type: ignore[arg-type]

    with unittest.mock.patch.object(ect, "_on_session_close", lambda ev: None):
        ect._delayed_session_shutdown = _spy_delayed_shutdown  # type: ignore[method-assign]
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(GoodbyeAgent())

            session.generate_reply()
            while not model.active_session._reply_futs:
                await asyncio.sleep(0)
            model.active_session._reply_futs[0].set_result(
                _generation(
                    response_id="goodbye",
                    text="goodbye!",
                    audio_duration=0.3,
                    function_calls=[FunctionCall(call_id="1", name="end_call", arguments="{}")],
                )
            )
            # playout + shutdown decision, without waiting out the 5s reply timeout
            await asyncio.sleep(1.0)

    assert not delayed_wait_started.is_set(), (
        "no tool-reply wait may be armed when the goodbye was in the same turn"
    )
    assert session._is_closing(), "session should shut down right after the goodbye"


async def test_silent_end_call_still_waits_for_tool_reply() -> None:
    """No assistant message in the turn: the tool reply IS the goodbye, keep the wait."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    ect = EndCallTool(end_instructions=None)

    class QuietAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test", tools=[ect])

    with unittest.mock.patch.object(ect, "_on_session_close", lambda ev: None):
        closed = asyncio.Event()
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            session.once("close", lambda *_: closed.set())
            await session.start(QuietAgent())

            session.generate_reply()
            while not model.active_session._reply_futs:
                await asyncio.sleep(0)
            model.active_session._reply_futs[0].set_result(
                _generation(
                    response_id="silent",
                    text="",
                    audio_duration=0.1,
                    function_calls=[FunctionCall(call_id="1", name="end_call", arguments="{}")],
                )
            )
            await asyncio.sleep(0.8)

            # the reply-wait path was taken: the delayed shutdown task exists and is still
            # waiting for the tool-reply speech (nothing closed the session early)
            task = ect._shutdown_session_task
            assert task is not None and not task.done()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            session.shutdown()
            await asyncio.wait_for(closed.wait(), timeout=5.0)
