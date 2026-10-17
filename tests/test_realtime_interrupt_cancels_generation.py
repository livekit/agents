"""Interrupting a realtime speech must cancel the response the model is still generating.

Every interrupt path counts, not only ``AgentActivity.interrupt()``: a bare
``SpeechHandle.interrupt()`` releases ``wait_for_playout()``, so it has to reach the provider
too. A pause is not an interruption, so a paused playout must leave the response alone. The
cancel is session-wide, so a speech whose generation already finished must never send one:
it would stop the reply that took its place.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm, utils
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput
from .fake_realtime import (
    FakeRealtimeModel,
    FakeRealtimeSession,
    _audio_frame,
    fake_capabilities,
)

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@asynccontextmanager
async def _speaking_reply() -> AsyncIterator[
    tuple[
        AgentSession,
        FakeRealtimeSession,
        SpeechHandle,
        FakeAudioOutput,
        utils.aio.Chan[llm.FunctionCall],
    ]
]:
    """Drive a realtime reply whose second of audio is still playing when the test resumes.

    The function stream is yielded still open, so the response ends when the test closes it.
    """
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    audio_output = FakeAudioOutput(can_pause=True)

    async with AgentSession(llm=model) as session:
        session.output.audio = audio_output
        await session.start(Agent(instructions="test"))

        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("a message the app cuts short")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()

        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        while audio_output._started_at is None:
            await asyncio.sleep(0)

        yield session, model.active_session, handle, audio_output, function_ch


async def test_forced_speech_interrupt_cancels_the_response() -> None:
    async with _speaking_reply() as (_, rt_session, handle, _, function_ch):
        handle.interrupt(force=True)
        await asyncio.sleep(0.1)

        assert rt_session.interrupted

        function_ch.close()
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)


async def test_interrupting_a_played_out_response_spares_the_newer_reply() -> None:
    # the model finishes the response long before the buffered audio does, so from here on a
    # cancel would land on whichever response the model started next
    async with _speaking_reply() as (_, rt_session, handle, _, function_ch):
        function_ch.close()
        await asyncio.sleep(0.1)

        handle.interrupt(force=True)
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

        assert handle.interrupted
        assert not rt_session.interrupted


async def test_interrupting_a_queued_finished_response_spares_the_newer_reply() -> None:
    # a server-initiated turn waits for the speech ahead of it to play out, and its own
    # response can end, and be replaced, long before it is ever authorized to speak
    async with _speaking_reply() as (session, rt_session, _, _, function_ch):
        queued: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: queued.append(ev.speech_handle))

        b_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        b_function_ch = utils.aio.Chan[llm.FunctionCall]()
        b_message_ch.close()
        b_function_ch.close()
        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=b_message_ch,
                function_stream=b_function_ch,
                user_initiated=False,
                response_id="response-b",
            ),
        )
        await asyncio.sleep(0.1)

        assert len(queued) == 1
        queued[0].interrupt(force=True)
        await asyncio.sleep(0.1)

        assert queued[0].interrupted
        assert not rt_session.interrupted

        function_ch.close()


async def test_paused_playout_keeps_the_response() -> None:
    async with _speaking_reply() as (_, rt_session, handle, audio_output, function_ch):
        function_ch.close()
        audio_output.pause()
        await asyncio.sleep(1.0)

        assert not handle.interrupted
        assert not rt_session.interrupted

        audio_output.resume()
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

        assert not rt_session.interrupted


async def test_interrupt_racing_the_generation_event_cancels_the_response() -> None:
    # the generation event and the interrupt land in the same tick, so the reply task returns
    # before it ever forwards the response: nothing else is left to cancel it
    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))

        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        message_ch.close()
        function_ch.close()
        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )
        handle.interrupt(force=True)
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

        assert model.active_session.interrupted


async def test_interrupting_a_finished_speech_spares_the_newer_reply() -> None:
    # the older handle stays alive for its tool while the next reply generates, and the cancel
    # it would send is the one that reply is running on
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            tool_started.set()
            await release_tool.wait()
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        await session.start(ToolAgent())
        rt_session = model.active_session

        handle_a = session.generate_reply()
        while not rt_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-a",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("let me check")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(0.2))
        audio_ch.close()
        function_ch.send_nowait(llm.FunctionCall(call_id="1", name="lookup", arguments="{}"))
        function_ch.close()

        rt_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-a",
            )
        )

        await asyncio.wait_for(tool_started.wait(), timeout=5)
        handle_b = session.generate_reply()

        # A's generation ends with its playout, and only then is B authorized to ask for a response
        async def _wait_for_second_reply() -> None:
            while rt_session.generate_reply_calls < 2:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait_for_second_reply(), timeout=5)
        assert not handle_a.done()
        assert session._activity is not None
        assert handle_a in session._activity._background_speeches

        rt_session.interrupted = False
        handle_a.interrupt(force=True)
        await asyncio.sleep(0.1)

        assert handle_a.interrupted
        assert not rt_session.interrupted

        release_tool.set()
        b_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        b_function_ch = utils.aio.Chan[llm.FunctionCall]()
        b_message_ch.close()
        b_function_ch.close()
        rt_session._reply_futs[1].set_result(
            llm.GenerationCreatedEvent(
                message_stream=b_message_ch,
                function_stream=b_function_ch,
                user_initiated=True,
                response_id="response-b",
            )
        )

        await asyncio.wait_for(handle_b.wait_for_playout(), timeout=5)
        await asyncio.wait_for(handle_a.wait_for_playout(), timeout=5)

        assert not rt_session.interrupted
