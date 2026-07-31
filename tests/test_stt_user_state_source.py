"""Regression tests for #5580: with STT-driven turn detection, VAD must not drive user_state.

When ``turn_detection="stt"``, both VAD and STT used to write ``user_state``.
In noisy environments VAD flips it to "speaking" on background noise even when
the STT hears nothing, breaking everything keyed on user state (away timeouts,
filler triggering) — and the only workaround, ``vad=None``, also gave up
VAD-based interruption sensing. STT is now the authoritative ``user_state``
source in that mode, while VAD keeps its interruption/endpointing roles.
"""

import time

import pytest

from livekit.agents import vad
from livekit.agents.voice.agent_activity import AgentActivity

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
