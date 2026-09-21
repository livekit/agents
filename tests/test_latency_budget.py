import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    LatencyBudgetEvent,
    LatencyBudgetOptions,
    llm,
    utils,
)
from livekit.agents.voice.agent_activity import _UserTurnCompletedContextVar, _UserTurnCompletedData

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, _audio_frame

pytestmark = pytest.mark.unit


def test_latency_budget_options_validation() -> None:
    with pytest.raises(ValueError, match="budget.*greater than zero"):
        AgentSession(latency_budget={"budget": 0})

    with pytest.raises(ValueError, match="warning.*no greater than budget"):
        AgentSession(latency_budget={"budget": 1.0, "warning": 1.1})


@pytest.mark.parametrize(
    ("options", "threshold"),
    [
        ({"budget": float("nan")}, "budget"),
        ({"budget": float("inf")}, "budget"),
        ({"budget": float("-inf")}, "budget"),
        ({"budget": 1.0, "warning": float("nan")}, "warning"),
        ({"budget": 1.0, "warning": float("inf")}, "warning"),
        ({"budget": 1.0, "warning": float("-inf")}, "warning"),
    ],
)
def test_latency_budget_rejects_non_finite_thresholds(
    options: LatencyBudgetOptions, threshold: str
) -> None:
    with pytest.raises(ValueError, match=rf"{threshold}.*finite"):
        AgentSession(latency_budget=options)


@pytest.mark.parametrize(
    ("latency", "expected_level", "expected_threshold"),
    [(0.8, "warning", 0.5), (1.2, "exceeded", 1.0)],
)
def test_latency_budget_event(
    latency: float, expected_level: str, expected_threshold: float
) -> None:
    session = AgentSession(latency_budget={"budget": 1.0, "warning": 0.5})
    events: list[LatencyBudgetEvent] = []
    session.on("latency_budget", events.append)

    session._evaluate_latency_budget(latency=latency, speech_id="speech-1")

    assert len(events) == 1
    assert events[0].level == expected_level
    assert events[0].threshold == expected_threshold
    assert events[0].budget == 1.0
    assert events[0].latency == latency
    assert events[0].speech_id == "speech-1"


def test_latency_budget_does_not_emit_within_warning_threshold() -> None:
    session = AgentSession(latency_budget={"budget": 1.0, "warning": 0.5})
    events: list[LatencyBudgetEvent] = []
    session.on("latency_budget", events.append)

    session._evaluate_latency_budget(latency=0.49, speech_id="speech-1")

    assert events == []


@pytest.mark.parametrize("interrupted", [False, True])
async def test_realtime_stalled_turn_reaches_threshold_without_output(
    interrupted: bool,
) -> None:
    model = FakeRealtimeModel()
    events: list[LatencyBudgetEvent] = []
    async with AgentSession(llm=model, latency_budget={"budget": 0.02}) as session:
        session.on("latency_budget", events.append)
        await session.start(Agent(instructions="test"))
        model.active_session.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        if interrupted:
            model.active_session.emit("input_speech_started", llm.InputSpeechStartedEvent())
        await asyncio.sleep(0.04)

    assert len(events) == (0 if interrupted else 1)
    if not interrupted:
        assert events[0].level == "exceeded"
        assert events[0].speech_id is None
        assert events[0].latency >= 0.02


async def test_stalled_turn_emits_warning_then_exceeded() -> None:
    model = FakeRealtimeModel()
    events: list[LatencyBudgetEvent] = []
    async with AgentSession(llm=model, latency_budget={"warning": 0.01, "budget": 0.03}) as session:
        session.on("latency_budget", events.append)
        await session.start(Agent(instructions="test"))
        model.active_session.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        await asyncio.sleep(0.05)

    assert [event.level for event in events] == ["warning", "exceeded"]
    assert [event.speech_id for event in events] == [None, None]


@pytest.mark.parametrize("provider_order", ["stop_first", "google", "ultravox"])
async def test_realtime_server_turn_emits_latency_budget_on_first_output(
    provider_order: str,
) -> None:
    model = FakeRealtimeModel()
    events: list[LatencyBudgetEvent] = []
    event_received = asyncio.Event()

    async with AgentSession(llm=model, latency_budget={"budget": 0.001}) as session:
        session.output.audio = FakeAudioOutput()

        def _on_latency_budget(event: LatencyBudgetEvent) -> None:
            events.append(event)
            event_received.set()

        session.on("latency_budget", _on_latency_budget)
        await session.start(Agent(instructions="test"))

        rt_session = model.active_session
        if provider_order == "stop_first":
            rt_session.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
            )
        await asyncio.sleep(0.01)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
            ),
        )

        if provider_order == "ultravox":
            rt_session.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
            )

        # Keep the first output beyond the 1 ms budget. On fast CI runners an
        # immediately queued frame can start within budget, correctly producing no alert.
        await asyncio.sleep(0.01)
        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        function_ch.close()
        text_ch.send_nowait("Hello")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(0.01))
        audio_ch.close()

        await asyncio.wait_for(event_received.wait(), timeout=5)
        if provider_order == "google":
            rt_session.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
            )

    assert len(events) == 1
    assert events[0].level == "exceeded"
    assert events[0].latency >= 0.001
    if provider_order != "stop_first":
        assert events[0].speech_id


@pytest.mark.parametrize("provider_order", ["stop_first", "google"])
async def test_realtime_auto_tool_reply_does_not_start_another_latency_turn(
    provider_order: str,
) -> None:
    model = FakeRealtimeModel()
    events: list[LatencyBudgetEvent] = []
    speeches = []

    async with AgentSession(llm=model, latency_budget={"budget": 0.02}) as session:
        session.output.audio = FakeAudioOutput()
        session.on("latency_budget", events.append)
        session.on("speech_created", speeches.append)
        await session.start(Agent(instructions="test"))
        rt_session = model.active_session

        if provider_order == "stop_first":
            rt_session.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
            )

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        function_ch.close()
        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
            ),
        )
        await asyncio.sleep(0.03)
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="first-output",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("Let me check")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(0.01))
        audio_ch.close()
        await asyncio.wait_for(speeches[0].speech_handle.wait_for_playout(), timeout=5)

        if provider_order == "google":
            rt_session.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
            )

        # The server's automatic post-tool generation belongs to the same user turn.
        tool_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        tool_function_ch = utils.aio.Chan[llm.FunctionCall]()
        tool_message_ch.close()
        tool_function_ch.close()
        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=tool_message_ch,
                function_stream=tool_function_ch,
                user_initiated=False,
            ),
        )
        await asyncio.sleep(0.04)
        assert len(events) == 1

        # A genuinely new user stop must still arm the next latency watch.
        rt_session.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
        )
        await asyncio.sleep(0.04)
        assert len(events) == 2


@pytest.mark.parametrize("spawned_task", [False, True])
async def test_realtime_hook_say_emits_latency_budget_on_first_output(
    spawned_task: bool,
) -> None:
    model = FakeRealtimeModel()
    events: list[LatencyBudgetEvent] = []

    async with AgentSession(llm=model, latency_budget={"budget": 0.001}) as session:
        session.output.audio = FakeAudioOutput()
        session.on("latency_budget", events.append)
        await session.start(Agent(instructions="test"))

        assert session._activity is not None
        stopped_at = time.time() - 0.01
        context = _UserTurnCompletedContextVar.set(
            _UserTurnCompletedData(
                activity=session._activity,
                metrics={"stopped_speaking_at": stopped_at},
                task=asyncio.current_task(),
            )
        )
        try:
            if not spawned_task:
                session._start_latency_budget_watch(stopped_at)
            if spawned_task:

                async def _background_say():
                    return session.say("Hello")

                speech = await asyncio.create_task(_background_say())
            else:
                speech = session.say("Hello")
        finally:
            _UserTurnCompletedContextVar.reset(context)

        while not model.active_session.say_futs:
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
        function_ch.close()
        text_ch.send_nowait("Hello")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(0.01))
        audio_ch.close()
        model.active_session.say_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
            )
        )
        await asyncio.wait_for(speech.wait_for_playout(), timeout=5)

    assert len(events) == (0 if spawned_task else 1)
    if not spawned_task:
        assert events[0].speech_id == speech.id
        assert events[0].level == "exceeded"
