"""AMD's predictions and completion reach the remote session host (spec §13 on ``main``): what an
``lk agent console`` client is sent during an outbound call under AMD."""

from __future__ import annotations

import pytest

from livekit.agents import AMD, Agent, AMDCompletedEvent, AMDReason
from livekit.protocol.agent_pb.agent_session import AgentSessionEvent, AmdCategory
from tests.fakes import until
from tests.host.test_amd import AGENT_INSTRUCTIONS, Classifier, voice_session

from .conftest import ConsolePeer


def forwarded(console: ConsolePeer, kind: str) -> list[AgentSessionEvent]:
    return [
        m.event
        for m in console.messages
        if m.WhichOneof("message") == "event" and m.event.WhichOneof("event") == kind
    ]


@pytest.mark.usefixtures("no_logged_tracebacks")
async def test_screening_prediction_is_forwarded_to_session_observability(
    console: ConsolePeer,
) -> None:
    """Each prediction reaches the client with its turn, stage and model time."""
    greeting = "Please state your name and why you are calling."
    session, microphone = voice_session([greeting])
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        session.input.audio = microphone
        classifier = Classifier({greeting: "machine-screening"})
        async with AMD(session, llm=classifier, stt=None, machine_silence_threshold=0):
            microphone.speak(0.3)
            await until(lambda: forwarded(console, "amd_prediction"), "no prediction forwarded")
            [event] = forwarded(console, "amd_prediction")
            prediction = event.amd_prediction
            assert prediction.category == AmdCategory.AMD_MACHINE_SCREENING
            assert prediction.stage == AmdCategory.AMD_MACHINE_SCREENING
            assert prediction.turn_id == 1
            assert prediction.reason == AMDReason.PREDICTION
            assert prediction.transcript == greeting
            assert prediction.HasField("inference_duration")


@pytest.mark.parametrize("listener", ["ok", "failing"])
async def test_completion_is_forwarded_to_session_observability(
    console: ConsolePeer, listener: str
) -> None:
    """The completion reaches the client once with its last prediction, even when an
    ``amd_completed`` listener raises (§12.3)."""
    greeting = "Hello, who is this?"
    session, microphone = voice_session([greeting])
    completed: list[AMDCompletedEvent] = []

    def on_completed(event: AMDCompletedEvent) -> None:
        completed.append(event)
        if listener == "failing":
            raise TypeError("listener bug")

    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        session.input.audio = microphone
        detector = AMD(session, llm=Classifier({greeting: "human"}), stt=None)
        detector.on("amd_completed", on_completed)
        async with detector:
            microphone.speak(0.3)
            result = await detector.execute()
        await until(lambda: forwarded(console, "amd_completed"), "no completion forwarded")
    assert completed == [result]
    [event] = forwarded(console, "amd_completed")
    assert event.amd_completed.category == AmdCategory.AMD_HUMAN
    assert event.amd_completed.reason == AMDReason.FINISHED
    assert not event.amd_completed.voicemail_message_played
    assert event.amd_completed.HasField("prediction")
    assert event.amd_completed.prediction.turn_id == 1
    assert event.amd_completed.prediction.category == AmdCategory.AMD_HUMAN


@pytest.mark.usefixtures("no_logged_tracebacks")
async def test_a_completion_without_a_prediction_is_forwarded_without_one(
    console: ConsolePeer,
) -> None:
    """A run closed before any prediction forwards its completion with the prediction unset
    (§13)."""
    session, _ = voice_session([])
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        async with AMD(session, llm=Classifier(), stt=None) as detector:
            pass
        assert (await detector.execute()).reason == AMDReason.CANCELLED
        await until(lambda: forwarded(console, "amd_completed"), "no completion forwarded")
    [event] = forwarded(console, "amd_completed")
    assert event.amd_completed.category == AmdCategory.AMD_UNCERTAIN
    assert event.amd_completed.reason == AMDReason.CANCELLED
    assert not event.amd_completed.HasField("prediction")
