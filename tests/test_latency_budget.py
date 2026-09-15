import asyncio

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


async def test_realtime_server_turn_emits_latency_budget_on_first_output() -> None:
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
        rt_session.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        await asyncio.sleep(0.01)

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

        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
            ),
        )

        await asyncio.wait_for(event_received.wait(), timeout=5)

    assert len(events) == 1
    assert events[0].level == "exceeded"
    assert events[0].latency >= 0.001
    assert events[0].speech_id
