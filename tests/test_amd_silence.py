from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from livekit.agents import AMD, AgentSession, llm, stt, vad
from livekit.agents.voice.amd import AMDCategory, detector as detector_module

from .test_amd_detector import commit, end_of_turn, eventually, running

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@pytest.fixture(autouse=True)
def detector_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "time",
        SimpleNamespace(monotonic=lambda: asyncio.get_running_loop().time()),
    )


def test_machine_silence_default_and_validation() -> None:
    assert AMD(AgentSession())._machine_silence_threshold == 1.5
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
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0.5)
        await commit(detector, session, classifier, reply=True)
        await asyncio.sleep(0.2)
        classifier.prediction(1, category)
        await asyncio.sleep(0.7)
        assert events == []
        assert model.calls.empty()
        assert detector._category == AMDCategory.UNCERTAIN
        assert not detector._turns[1].decision.done()
        assert not detector._turns[1].timed_out
        assert detector._inference_timeouts == 0
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
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0)
        info = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        assert await detector._prepare_reply(info, llm.ChatContext())
        assert detector._latest.category == category
        assert detector._latest.delay < 0.01


@pytest.mark.asyncio
async def test_elapsed_silence_and_slow_inference_do_not_add_another_wait() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0.5)
        info = await commit(detector, session, classifier)
        await asyncio.sleep(1.2)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await detector._prepare_reply(info, llm.ChatContext())
        assert detector._latest.delay == pytest.approx(1.2, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["prediction", "inference_timeout", "inference_error", "reused"])
async def test_established_machine_stage_gates_uncertain_and_fallbacks(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        detector._on_user_speech_started()
        detector._on_user_speech_ended(1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await detector._prepare_reply(first, llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0.5)
        if reason == "reused":
            detector._on_end_of_turn(end_of_turn(""))
        else:
            await commit(detector, session, classifier)
            if reason == "prediction":
                classifier.prediction(2, AMDCategory.UNCERTAIN)
            elif reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        await asyncio.sleep(0.9)
        assert events == []
        assert not detector._turns[2].decision.done()
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
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0.5)
        info = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.sleep(0.5)
        detector._on_user_speech_started()
        await asyncio.sleep(1)
        detector._on_user_speech_ended(0.5)
        await asyncio.sleep(1.1)
        assert events == []
        assert not detector._turns[1].decision.done()
        detector._on_end_of_turn(end_of_turn("Hello, can you hear me?"))
        request = await classifier.request()
        assert request.earlier_turns[0]["transcript"] == "hello"
        assert not await detector._prepare_reply(info, llm.ChatContext())
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
        detector._on_user_speech_ended(1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await detector._prepare_reply(first, llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0)
        await commit(detector, session, classifier)
        await asyncio.sleep(0.3)
        assert detector._turns[2].timed_out
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
        detector._on_user_speech_ended(1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._prepare_reply(first, llm.ChatContext())
        for turn_id in range(2, 5):
            detector._on_user_speech_started()
            detector._on_user_speech_ended(0.5)
            await commit(detector, session, classifier)
            await asyncio.sleep(0.3)
            assert detector.enabled
            assert not detector._turns[turn_id].decision.done()
            await asyncio.sleep(0.71)
            assert detector._turns[turn_id].decision.done()
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
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0.5)
        await commit(detector, session, classifier)
        if inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.2)
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0.5)
        detector._on_end_of_turn(end_of_turn(""))
        if not inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.9)
        assert events == []
        await asyncio.sleep(0.11)
        assert [(e.turn_id, e.reason) for e in events] == [(1, "prediction"), (2, "reused")]
        assert detector._turns[2].decision.done()
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_new_inference_settles_queued_empty_turns() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        detector._on_user_speech_ended(1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._prepare_reply(first, llm.ChatContext())
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0)
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.1)
        detector._on_end_of_turn(end_of_turn(""))
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0)
        fourth = await commit(detector, session, classifier)
        assert detector._turns[2].decision.done()
        assert detector._turns[3].decision.done()
        classifier.prediction(4, AMDCategory.MACHINE_SCREENING)
        await detector._prepare_reply(fourth, llm.ChatContext())
        assert detector._idle_timer is not None


@pytest.mark.asyncio
async def test_prediction_listener_failure_cleans_up_a_deferred_release() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):

        def on_prediction(_event: object) -> None:
            raise TypeError("listener failed")

        detector.on("amd_prediction", on_prediction)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert (await detector.execute()).reason == "inference_error"
        assert detector._turns[1].decision.done()
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
        await detector._prepare_reply(info, llm.ChatContext())
        assert detector._idle_timer is not None
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.2)
        detector._rearm_idle()
        assert detector._idle_timer is None
        assert detector.enabled
        await asyncio.sleep(1.31)
        assert detector._latest.category == AMDCategory.MACHINE_SCREENING
        assert detector._idle_timer is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [None, 0.7])
async def test_eot_without_speech_edges_uses_available_timing(elapsed: float | None) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, _, classifier, _):
        info = end_of_turn()
        info.metrics.end_of_turn_delay = elapsed
        detector._on_end_of_turn(info)
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._turns[1].decision
        assert detector._latest.delay == pytest.approx(1.5 - (elapsed or 0), abs=0.01)


@pytest.mark.asyncio
async def test_eot_before_speech_end_waits_for_the_speech_end_anchor() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        detector._on_user_speech_started()
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.6)
        assert not detector._turns[1].decision.done()
        detector._on_user_speech_ended(0.5)
        await detector._turns[1].decision
        assert detector._latest.delay == pytest.approx(1.6, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt"])
async def test_audio_recognition_supplies_speech_edges_and_elapsed_silence(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, _, _):
        recognition = session._activity._audio_recognition
        monkeypatch.setattr(recognition, "_run_eou_detection", Mock())
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
            monkeypatch.setattr(recognition, "_vad", None)
            recognition._turn_detection_mode = "stt"
            recognition._process_stt_event(
                stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            )
        assert detector._speaking_since is not None
        assert detector._speech_epoch == 1
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
        assert detector._speaking_since is None
        assert detector._speech_ended_at == pytest.approx(
            asyncio.get_running_loop().time() - 0.7, abs=0.01
        )


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
        await eventually(lambda: detector._turns[1].release_timer is not None)
        timer = detector._turns[1].release_timer
        if reason == "cancelled":
            await detector.aclose()
        elif reason == "participant_disconnected":
            detector._participant_identity = "callee"
            detector._on_disconnected(SimpleNamespace(identity="callee"))
        assert (await detector.execute()).reason == reason
        assert timer.cancelled()
        assert detector._turns[1].decision.done()
        assert events == []
        assert session.amd is None
        assert not detector._tasks
