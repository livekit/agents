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
