"""generate_reply() inside a realtime tool must not beat function_call_output (#7915).

Gemini Live cancels a response.create that arrives while a function call is still
open. Models that advertise generate_reply_during_tool_call=False defer that reply
until tool outputs have been pushed; awaiting the deferred handle from the tool
raises a clear circular-wait error.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, utils
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _text_generation(
    *,
    response_id: str,
    text: str,
    message_id: str,
    function_calls: list[FunctionCall] | None = None,
) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["text"])

    message_ch.send_nowait(
        MessageGeneration(
            message_id=message_id,
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    text_ch.send_nowait(text)
    text_ch.close()
    audio_ch.close()
    for fnc in function_calls or []:
        function_ch.send_nowait(fnc)
    function_ch.close()

    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=True,
        response_id=response_id,
    )


def _tool_generation(*, tool_name: str) -> GenerationCreatedEvent:
    return _text_generation(
        response_id="response-id",
        text="let me check",
        message_id="message-id",
        function_calls=[FunctionCall(call_id="1", name=tool_name, arguments="{}")],
    )


async def _wait_for_reply_fut(model: FakeRealtimeModel, *, min_count: int = 1) -> None:
    for _ in range(500):
        if len(model.active_session._reply_futs) >= min_count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected >= {min_count} generate_reply futures")


@pytest.mark.parametrize("during_tool_ok", [True, False])
async def test_generate_reply_in_tool_ordering(during_tool_ok: bool) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            audio_output=False,
            auto_tool_reply_generation=False,
            generate_reply_during_tool_call=during_tool_ok,
        )
    )
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def next_topic(self) -> str:
            """Move on to the next topic."""
            tool_started.set()
            self.session.generate_reply(instructions="Introduce cafes.")
            await release_tool.wait()
            return "ok"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())

        speech = session.generate_reply()
        await _wait_for_reply_fut(model, min_count=1)
        model.active_session._reply_futs[0].set_result(_tool_generation(tool_name="next_topic"))

        await asyncio.wait_for(tool_started.wait(), timeout=5)

        # while the tool is still running, a deferred model must not have issued
        # the in-tool response.create yet
        if not during_tool_ok:
            assert model.active_session.generate_reply_calls == 1
            assert "function_call_output" not in model.active_session.event_log

        release_tool.set()

        # complete any further generate_reply futures the session opens
        async def _resolve_pending() -> None:
            seen = 1
            for _ in range(500):
                futs = model.active_session._reply_futs
                while seen < len(futs):
                    fut = futs[seen]
                    if not fut.done():
                        fut.set_result(
                            _text_generation(
                                response_id=f"response-{seen}",
                                text="next topic",
                                message_id=f"followup-{seen}",
                            )
                        )
                    seen += 1
                if speech.done():
                    return
                await asyncio.sleep(0)
            raise AssertionError("speech did not finish")

        await asyncio.wait_for(_resolve_pending(), timeout=10)
        await asyncio.wait_for(speech.wait_for_playout(), timeout=5)

    events = model.active_session.event_log
    assert "function_call_output" in events
    assert events.count("generate_reply") >= 2
    first_output = events.index("function_call_output")
    # the second generate_reply is the one issued from inside the tool (or the
    # automatic tool reply). with deferral it must follow the tool output.
    second_reply = [i for i, e in enumerate(events) if e == "generate_reply"][1]
    if during_tool_ok:
        assert second_reply < first_output
    else:
        assert second_reply > first_output


async def test_awaiting_deferred_generate_reply_in_tool_raises() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            audio_output=False,
            auto_tool_reply_generation=False,
            generate_reply_during_tool_call=False,
        )
    )
    saw_error = asyncio.Event()
    error_text: list[str] = []

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def next_topic(self) -> str:
            """Move on to the next topic."""
            handle = self.session.generate_reply(instructions="Introduce cafes.")
            try:
                await handle.wait_for_playout()
            except RuntimeError as e:
                error_text.append(str(e))
                saw_error.set()
            return "ok"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())

        speech = session.generate_reply()
        await _wait_for_reply_fut(model, min_count=1)
        model.active_session._reply_futs[0].set_result(_tool_generation(tool_name="next_topic"))

        await asyncio.wait_for(saw_error.wait(), timeout=5)
        assert error_text
        assert "deferred" in error_text[0].lower() or "deadlock" in error_text[0].lower()

        # resolve follow-up replies so the session can drain
        for _ in range(500):
            for fut in model.active_session._reply_futs[1:]:
                if fut.done():
                    continue
                fut.set_result(
                    _text_generation(
                        response_id="followup",
                        text="ok",
                        message_id="followup",
                    )
                )
            if speech.done():
                break
            await asyncio.sleep(0)

        await asyncio.wait_for(speech.wait_for_playout(), timeout=5)


def _deferring_model() -> FakeRealtimeModel:
    return FakeRealtimeModel(
        capabilities=fake_capabilities(
            audio_output=False,
            auto_tool_reply_generation=False,
            generate_reply_during_tool_call=False,
        )
    )


async def _drain(model: FakeRealtimeModel, speech) -> None:  # type: ignore[no-untyped-def]
    for _ in range(500):
        for fut in model.active_session._reply_futs[1:]:
            if not fut.done():
                fut.set_result(_text_generation(response_id="f", text="ok", message_id="f"))
        if speech.done():
            return
        await asyncio.sleep(0)


async def test_interrupt_cancels_deferred_generate_reply_in_tool() -> None:
    model = _deferring_model()
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    handles: list = []

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def next_topic(self) -> str:
            """Move on to the next topic."""
            handles.append(self.session.generate_reply(instructions="Introduce cafes."))
            tool_started.set()
            await release_tool.wait()
            return "ok"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())

        speech = session.generate_reply()
        await _wait_for_reply_fut(model, min_count=1)
        model.active_session._reply_futs[0].set_result(_tool_generation(tool_name="next_topic"))
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        session.interrupt(force=True)
        assert handles[0].interrupted
        assert not session._activity._deferred_tool_replies  # type: ignore[union-attr]

        release_tool.set()
        await _drain(model, speech)

    assert handles[0].interrupted


async def test_failed_tool_output_sync_cancels_deferred_generate_reply() -> None:
    from livekit.agents.llm import RealtimeError

    model = _deferring_model()
    handles: list = []

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def next_topic(self) -> str:
            """Move on to the next topic."""
            handles.append(self.session.generate_reply(instructions="Introduce cafes."))
            model.active_session.update_error = RealtimeError("sync failed")
            return "ok"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())

        speech = session.generate_reply()
        await _wait_for_reply_fut(model, min_count=1)
        model.active_session._reply_futs[0].set_result(_tool_generation(tool_name="next_topic"))
        await _drain(model, speech)
        for _ in range(200):
            if handles and handles[0].done():
                break
            await asyncio.sleep(0)

        assert handles
        # the function call is still open server-side: the in-tool reply must not go out
        assert handles[0].interrupted
        assert "function_call_output" not in model.active_session.event_log
        model.active_session.update_error = None


async def test_rejected_interrupt_keeps_deferred_generate_reply_in_tool() -> None:
    """interrupt(force=False) on protected speech must not drop deferred replies."""
    model = _deferring_model()
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    handles: list = []

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def next_topic(self) -> str:
            """Move on to the next topic."""
            handles.append(self.session.generate_reply(instructions="Introduce cafes."))
            tool_started.set()
            await release_tool.wait()
            return "ok"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())

        speech = session.generate_reply()
        await _wait_for_reply_fut(model, min_count=1)
        # protect the active speech so a non-forced interrupt is rejected
        speech.allow_interruptions = False
        model.active_session._reply_futs[0].set_result(_tool_generation(tool_name="next_topic"))
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            session.interrupt(force=False)

        assert handles
        assert not handles[0].interrupted
        deferred = session._activity._deferred_tool_replies  # type: ignore[union-attr]
        assert deferred and deferred[0][1] is handles[0]

        release_tool.set()
        await _drain(model, speech)

    assert not handles[0].interrupted
