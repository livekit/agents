from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from livekit.agents import AMD, Agent, AgentSession, llm, stt, vad
from livekit.agents.voice.amd import AMDCategory, detector as detector_module

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD
from .test_amd_detector import (
    CustomerAgent,
    commit,
    commit_turn,
    end_of_turn,
    eventually,
    running,
    speech_ended,
    speech_started,
)
from .test_amd_local_inference import DrainingSTT, push_audio

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@pytest.fixture(autouse=True)
def detector_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "time",
        SimpleNamespace(monotonic=lambda: asyncio.get_running_loop().time(), time=time.time),
    )


def test_machine_silence_default_and_validation() -> None:
    assert AMD(AgentSession(), llm=None, stt=None)._fsm._machine_silence_threshold == 1.5
    with pytest.raises(ValueError, match="machine_silence_threshold"):
        AMD(AgentSession(), machine_silence_threshold=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
        AMDCategory.MACHINE_UNAVAILABLE,
    ],
)
async def test_machine_prediction_and_reply_wait_for_remaining_silence(
    category: AMDCategory,
) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.3) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        await asyncio.sleep(0.2)
        classifier.prediction(1, category)
        await asyncio.sleep(0.7)
        assert events == []
        assert model.calls.empty()
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        assert detector._fsm.decision(1) is None
        assert not detector._fsm._turns[1].timed_out
        assert detector._fsm._inference_timeouts == 0
        await asyncio.sleep(0.11)
        assert len(events) == 1
        assert events[0].category == category
        assert events[0].delay == pytest.approx(1.0, abs=0.01)
        assert events[0].inference_duration == pytest.approx(0.2, abs=0.01)
        if category == AMDCategory.MACHINE_UNAVAILABLE:
            assert (await detector.execute()).category == category
            assert model.calls.empty()
        else:
            assert not model.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.UNCERTAIN])
async def test_human_and_initial_uncertain_do_not_wait(category: AMDCategory) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        speech_ended(detector, 0)
        info = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        assert await detector.should_reply(info.turn_id, llm.ChatContext())
        assert detector._fsm._latest.category == category
        assert detector._fsm._latest.delay < 0.01


@pytest.mark.asyncio
async def test_elapsed_silence_and_slow_inference_do_not_add_another_wait() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 0.5)
        info = await commit(detector, session, classifier)
        await asyncio.sleep(1.2)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await detector.should_reply(info.turn_id, llm.ChatContext())
        assert detector._fsm._latest.delay == pytest.approx(1.2, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["prediction", "inference_timeout", "inference_error", "reused"])
async def test_established_machine_stage_gates_uncertain_and_fallbacks(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await detector.should_reply(first.turn_id, llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        if reason == "reused":
            commit_turn(detector, end_of_turn(""))
        else:
            await commit(detector, session, classifier)
            if reason == "prediction":
                classifier.prediction(2, AMDCategory.UNCERTAIN)
            elif reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        await asyncio.sleep(0.9)
        assert events == []
        assert detector._fsm.decision(2) is None
        await asyncio.sleep(0.11)
        assert len(events) == 1
        assert events[0].category == AMDCategory.MACHINE_SCREENING
        assert events[0].reason == reason
        assert not events[0].state_changed
        assert events[0].delay == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_resumed_speech_invalidates_release_and_preserves_history() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        info = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.sleep(0.5)
        speech_started(detector)
        await asyncio.sleep(1)
        speech_ended(detector, 0.5)
        await asyncio.sleep(1.1)
        assert events == []
        assert detector._fsm.decision(1) is None
        commit_turn(detector, end_of_turn("Hello, can you hear me?"))
        request = await classifier.request()
        assert request.earlier_turns[0]["transcript"] == "hello"
        assert not await detector.should_reply(info.turn_id, llm.ChatContext())
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        assert not any(event.is_machine for event in events)


@pytest.mark.asyncio
async def test_late_human_prediction_replaces_a_machine_timeout_wait() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await detector.should_reply(first.turn_id, llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0)
        await commit(detector, session, classifier)
        await asyncio.sleep(0.3)
        assert detector._fsm._turns[2].timed_out
        assert not events
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        assert [event.category for event in events] == [AMDCategory.HUMAN]
        assert events[0].reason == "late_prediction"


@pytest.mark.asyncio
async def test_third_machine_timeout_waits_for_silence_before_completion() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector.should_reply(first.turn_id, llm.ChatContext())
        for turn_id in range(2, 5):
            speech_started(detector)
            speech_ended(detector, 0.5)
            await commit(detector, session, classifier)
            await asyncio.sleep(0.3)
            assert detector.enabled, (session._recorded_events, detector._fsm.completion())
            assert detector._fsm.decision(turn_id) is None
            await asyncio.sleep(0.71)
            assert detector._fsm.decision(turn_id) is not None
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("inference_ready", [True, False])
async def test_empty_eot_after_resumed_speech_rearms_a_useful_prediction(
    inference_ready: bool,
) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier)
        if inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.2)
        speech_started(detector)
        speech_ended(detector, 0.5)
        commit_turn(detector, end_of_turn(""))
        if not inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.9)
        assert events == []
        await asyncio.sleep(0.11)
        assert [(e.turn_id, e.reason) for e in events] == [(1, "prediction"), (2, "reused")]
        assert detector._fsm.decision(2) is not None
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_older_empty_transcript_wait_cannot_block_newer_prediction() -> None:
    stt = DrainingSTT()
    async with running(stt=stt, machine_silence_threshold=1.5) as (detector, _, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        stream = push_audio(detector, stt)
        speech_ended(detector, 0)
        commit_turn(detector, end_of_turn(""))
        await asyncio.wait_for(stream.flushed.wait(), 2)

        await asyncio.sleep(0.1)
        speech_started(detector)
        speech_ended(detector, 0)
        info = end_of_turn("Please state your name and why you are calling.")
        commit_turn(detector, info)
        assert (await classifier.request()).turn_id == 2
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)

        await asyncio.sleep(0.5)
        assert events == []
        await asyncio.sleep(1.01)
        assert [(event.turn_id, event.category) for event in events] == [
            (2, AMDCategory.MACHINE_SCREENING)
        ]
        assert detector._fsm.decision(1).reason == "superseded"
        assert await detector.should_reply(info.turn_id, llm.ChatContext())


@pytest.mark.asyncio
async def test_new_inference_settles_queued_empty_turns() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector.should_reply(first.turn_id, llm.ChatContext())
        speech_started(detector)
        speech_ended(detector, 0)
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.1)
        commit_turn(detector, end_of_turn(""))
        speech_started(detector)
        speech_ended(detector, 0)
        fourth = await commit(detector, session, classifier)
        assert detector._fsm.decision(2) is not None
        assert detector._fsm.decision(3) is not None
        classifier.prediction(4, AMDCategory.MACHINE_SCREENING)
        await detector.should_reply(fourth.turn_id, llm.ChatContext())
        assert detector._fsm._idle_deadline is not None


@pytest.mark.asyncio
async def test_prediction_listener_failure_cleans_up_a_deferred_release() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):

        def on_prediction(_event: object) -> None:
            raise TypeError("listener failed")

        detector.on("amd_prediction", on_prediction)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert (await detector.execute()).reason == "internal_error"
        assert detector._fsm.decision(1) is not None
        assert session.amd is None


@pytest.mark.asyncio
async def test_late_machine_prediction_suspends_idle_during_silence_wait() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.01, idle_timeout=0.1) as (
        detector,
        session,
        classifier,
        _,
    ):
        info = await commit(detector, session, classifier)
        await detector.should_reply(info.turn_id, llm.ChatContext())
        assert detector._fsm._idle_deadline is not None
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.2)
        detector._reschedule_timer()
        assert detector._fsm._idle_deadline is None
        assert detector.enabled
        await asyncio.sleep(1.31)
        assert detector._fsm._latest.category == AMDCategory.MACHINE_SCREENING
        assert detector._fsm._idle_deadline is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [None, 0.7])
async def test_eot_without_speech_edges_uses_available_timing(elapsed: float | None) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, _, classifier, _):
        info = end_of_turn()
        info.metrics.end_of_turn_delay = elapsed
        commit_turn(detector, info)
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._wait_for_decision(1)
        assert detector._fsm._latest.delay == pytest.approx(1.5 - (elapsed or 0), abs=0.01)


@pytest.mark.asyncio
async def test_eot_before_speech_end_waits_for_the_speech_end_anchor() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.6)
        assert detector._fsm.decision(1) is None
        speech_ended(detector, 0.5)
        await detector._wait_for_decision(1)
        assert detector._fsm._latest.delay == pytest.approx(1.6, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt", "stt_with_vad"])
async def test_audio_recognition_supplies_speech_edges_and_elapsed_silence(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, _, _):
        recognition = session._activity._audio_recognition
        monkeypatch.setattr(recognition, "_run_eou_detection", Mock())
        state_events = []
        session.on("user_state_changed", state_events.append)
        if source == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.1,
                    silence_duration=0,
                )
            )
        else:
            monkeypatch.setattr(
                recognition, "_vad", FakeVAD() if source == "stt_with_vad" else None
            )
            recognition._turn_detection_mode = "stt"
            recognition._process_stt_event(
                stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            )
        assert detector._fsm._speaking_since is not None
        assert detector._fsm._speech_epoch == 1
        await asyncio.sleep(1)
        if source == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.3,
                    silence_duration=0.5,
                    inference_duration=0.2,
                )
            )
        else:
            recognition._process_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, speech_end_time=time.time() - 0.7
                )
            )
        assert detector._fsm._speaking_since is None
        assert detector._fsm._speech_ended_at == pytest.approx(
            asyncio.get_running_loop().time() - 0.7, abs=0.01
        )
        assert state_events[-1].created_at == pytest.approx(time.time() - 0.7, abs=0.01)
        if source == "stt_with_vad":
            speech_ended_at = detector._fsm._speech_ended_at
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.3,
                    silence_duration=0.1,
                )
            )
            assert len(state_events) == 2
            assert detector._fsm._speech_ended_at == speech_ended_at
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.1,
                    silence_duration=0,
                )
            )
            assert detector._fsm._speaking_since is not None
            assert detector._fsm._speech_epoch == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt"])
async def test_untranscribed_speech_restarts_idle_timer(source: str) -> None:
    session = AgentSession(
        llm=FakeLLM(),
        stt=FakeSTT(),
        vad=FakeVAD() if source == "vad" else None,
        turn_handling={"turn_detection": "vad" if source == "vad" else "stt"},
    )
    await session.start(Agent(instructions="Answer the call."))
    try:
        async with AMD(session, llm=None, stt=None, idle_timeout=0.2, timeout=2) as detector:
            await eventually(lambda: detector.started)
            recognition = session._activity._audio_recognition
            if source == "vad":
                # Let the empty fake stream finish before injecting VAD events.
                await recognition._vad_atask
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.START_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            await asyncio.sleep(0.3)
            assert detector.enabled

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0.3,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
            await asyncio.sleep(0.1)
            assert detector.enabled
            result = await asyncio.wait_for(detector.execute(), 0.2)
            assert result.reason == "idle_timeout"
            assert result.turn_id == 0
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_speech_end_does_not_start_idle_while_customer_hook_runs() -> None:
    agent = CustomerAgent(stop=True)
    agent.hook_release.clear()
    async with running(agent=agent, idle_timeout=0.1) as (detector, session, classifier, _):
        await commit(detector, session, classifier, reply=True)
        await agent.hook_started.wait()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._wait_for_decision(1)
        speech_started(detector)
        speech_ended(detector, 0)
        await asyncio.sleep(0.2)
        assert detector.enabled

        agent.hook_release.set()
        result = await asyncio.wait_for(detector.execute(), 0.2)
        assert result.reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["timeout", "cancelled", "participant_disconnected"])
async def test_completion_cancels_silence_wait_and_releases_waiters(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, timeout=0.5) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await eventually(lambda: detector._fsm._turns[1].deadline is not None)
        timer = detector._timer
        if reason == "cancelled":
            await detector.aclose()
        elif reason == "participant_disconnected":
            detector._participant_identity = "callee"
            detector._on_disconnected(SimpleNamespace(identity="callee"))
        assert (await detector.execute()).reason == reason
        assert timer.cancelled()
        assert detector._fsm.decision(1) is not None
        assert events == []
        assert session.amd is None
        assert not detector._tasks
