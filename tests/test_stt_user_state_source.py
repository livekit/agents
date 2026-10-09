"""Regression tests for #5580: with STT-driven turn detection, VAD must not drive user_state.

When ``turn_detection="stt"``, both VAD and STT used to write ``user_state``.
In noisy environments VAD flips it to "speaking" on background noise even when
the STT hears nothing, breaking everything keyed on user state (away timeouts,
filler triggering) — and the only workaround, ``vad=None``, also gave up
VAD-based interruption sensing. STT is now the authoritative ``user_state``
source in that mode, while VAD keeps its interruption/endpointing roles.
"""

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit.agents import APIError, vad
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition, _STTPipeline
from livekit.agents.voice.endpointing import BaseEndpointing

from .fake_session import FakeActions, create_session
from .test_agent_session import MyAgent, _close_test_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _vad_event(type_: vad.VADEventType) -> vad.VADEvent:
    return vad.VADEvent(
        type=type_,
        samples_index=0,
        timestamp=time.time(),
        speech_duration=0.5,
        silence_duration=0.0,
    )


def _make_activity(turn_detection: str | None) -> AgentActivity:
    session = create_session(FakeActions(), turn_handling={"turn_detection": turn_detection})
    return AgentActivity(MyAgent(), session)


def _make_recognition(activity: AgentActivity, turn_detection: str) -> AudioRecognition:
    return AudioRecognition(
        activity._session,
        hooks=activity,
        endpointing=BaseEndpointing(min_delay=0.0, max_delay=0.0),
        stt=None,
        vad=None,
        using_default_vad=False,
        interruption_detection=None,
        turn_detection=turn_detection,
    )


class TestSttDrivenUserState:
    async def test_vad_noise_does_not_flip_user_state_in_stt_mode(self) -> None:
        activity = _make_activity("stt")
        session = activity._session
        try:
            assert session.user_state == "listening"

            # background noise: VAD fires but the STT hears no speech
            activity.on_start_of_speech(_vad_event(vad.VADEventType.START_OF_SPEECH), time.time())
            assert session.user_state == "listening"

            activity.on_end_of_speech(_vad_event(vad.VADEventType.END_OF_SPEECH))
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_stt_speech_drives_user_state_in_stt_mode(self) -> None:
        activity = _make_activity("stt")
        session = activity._session
        try:
            # STT-sourced hook calls pass ev=None
            activity.on_start_of_speech(None, time.time())
            assert session.user_state == "speaking"

            activity.on_end_of_speech(None)
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_vad_drives_user_state_in_default_mode(self) -> None:
        activity = _make_activity(None)
        session = activity._session
        try:
            activity.on_start_of_speech(_vad_event(vad.VADEventType.START_OF_SPEECH), time.time())
            assert session.user_state == "speaking"

            activity.on_end_of_speech(_vad_event(vad.VADEventType.END_OF_SPEECH))
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_vad_end_recovers_speaking_written_by_non_stt_source(self) -> None:
        # "speaking" can be entered by writers the STT will never clear
        # (claim_user_turn re-derivation, a turn_detection switch mid-speech);
        # a VAD end-of-speech must recover the state instead of leaving it
        # stuck at "speaking" with no STT end-of-speech ever coming
        activity = _make_activity("stt")
        session = activity._session
        try:
            session._update_user_state("speaking", last_speaking_time=time.time())
            assert session.user_state == "speaking"

            activity.on_end_of_speech(_vad_event(vad.VADEventType.END_OF_SPEECH))
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_vad_end_does_not_clear_stt_authored_speaking(self) -> None:
        # the VAD usually endpoints before the STT: its end-of-speech must not
        # cut short a "speaking" state the STT opened and will close itself
        activity = _make_activity("stt")
        session = activity._session
        try:
            activity.on_start_of_speech(None, time.time())
            assert session.user_state == "speaking"

            activity.on_end_of_speech(_vad_event(vad.VADEventType.END_OF_SPEECH))
            assert session.user_state == "speaking"

            activity.on_end_of_speech(None)
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_claimed_turn_with_vad_noise_recovers(self) -> None:
        # background noise trips the VAD during a programmatic (text) turn:
        # the release re-derives "speaking" from the VAD-driven silence event,
        # and only the later VAD end-of-speech can clear it (the STT heard
        # nothing, so no STT end-of-speech will ever arrive)
        activity = _make_activity("stt")
        session = activity._session
        session._activity = activity
        try:
            async with session._claim_user_turn():
                activity.on_start_of_speech(
                    _vad_event(vad.VADEventType.START_OF_SPEECH), time.time()
                )
            assert session.user_state == "speaking"

            activity.on_end_of_speech(_vad_event(vad.VADEventType.END_OF_SPEECH))
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_clear_user_turn_closes_open_stt_segment(self) -> None:
        # clear_user_turn() tears down and recreates the STT stream, so a
        # pending STT end-of-speech will never arrive; the open segment must
        # be closed on the way out or user_state stays "speaking" forever
        activity = _make_activity("stt")
        session = activity._session
        try:
            recognition = _make_recognition(activity, "stt")
            activity.on_start_of_speech(None, time.time())
            recognition._speaking = True  # as the STT START branch would have set
            assert session.user_state == "speaking"

            recognition._clear_user_turn()

            assert session.user_state == "listening"
            assert activity._stt_user_speaking is False
            assert recognition._speaking is False
        finally:
            await _close_test_session(session)

    async def test_clear_user_turn_leaves_state_to_vad_outside_stt_mode(self) -> None:
        # outside stt mode the VAD end-of-speech still owns the transition;
        # clearing the turn must not touch user_state
        activity = _make_activity(None)
        session = activity._session
        try:
            recognition = _make_recognition(activity, "vad")
            activity.on_start_of_speech(_vad_event(vad.VADEventType.START_OF_SPEECH), time.time())
            recognition._speaking = True
            assert session.user_state == "speaking"

            recognition._clear_user_turn()
            assert session.user_state == "speaking"

            activity.on_end_of_speech(_vad_event(vad.VADEventType.END_OF_SPEECH))
            assert session.user_state == "listening"
        finally:
            await _close_test_session(session)

    async def test_stt_reconnect_notifies_stream_reset(self) -> None:
        # a reconnect after an APIError drops the in-flight utterance's
        # END_OF_SPEECH with the old stream; the pipeline must notify its
        # owner so the open segment can be closed
        calls: list[int] = []

        def _failing_node(audio_ch: object, settings: object) -> AsyncIterable[object]:
            async def _gen() -> AsyncIterator[object]:
                raise APIError("connection dropped")
                yield  # unreachable; makes this an async generator

            return _gen()

        pipeline = _STTPipeline(_failing_node, on_stream_reset=lambda: calls.append(1))  # type: ignore[arg-type]
        try:
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0.01)
            assert calls, "on_stream_reset was not invoked on reconnect"
        finally:
            await pipeline.aclose()
