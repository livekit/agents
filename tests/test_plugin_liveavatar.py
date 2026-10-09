"""Unit tests for LiveAvatar WebSocket event dispatch.

The LiveAvatar LITE protocol emits ``agent.state_updated`` and
``agent.audio_buffer_*`` acks in addition to the original speak_* events.
These tests drive ``AvatarSession._handle_server_event`` with a fake audio
buffer so they stay hermetic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.plugins.liveavatar.avatar import AvatarSession

pytestmark = [pytest.mark.unit, pytest.mark.plugin("liveavatar")]


class _FakeAudioBuffer:
    def __init__(self) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []

    def notify_playback_started(self) -> None:
        self.started += 1

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self.finished.append((playback_position, interrupted))


@pytest.fixture
def avatar(monkeypatch: pytest.MonkeyPatch) -> tuple[AvatarSession, _FakeAudioBuffer]:
    def _fake_api(*_args: Any, **_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr("livekit.plugins.liveavatar.avatar.LiveAvatarAPI", _fake_api)
    session = AvatarSession(api_key="test-key", avatar_id="av-1")
    buf = _FakeAudioBuffer()
    session._audio_buffer = buf  # type: ignore[assignment]
    return session, buf


def test_session_connected_still_unblocks_forwarding(avatar):
    session, _ = avatar
    assert not session._session_connected.is_set()
    session._handle_server_event({"type": "session.state_updated", "state": "connected"})
    assert session._session_connected.is_set()


def test_state_updated_talking_marks_speaking_and_playback_started(avatar):
    session, buf = avatar
    session._handle_server_event(
        {
            "type": "agent.state_updated",
            "previous_state": "listening",
            "new_state": "talking",
        }
    )
    assert session._avatar_speaking is True
    assert buf.started == 1
    assert buf.finished == []


def test_state_updated_listening_finishes_playback(avatar):
    session, buf = avatar
    session._playback_position = 1.25
    session._handle_server_event(
        {"type": "agent.state_updated", "previous_state": "idle", "new_state": "talking"}
    )
    session._handle_server_event(
        {
            "type": "agent.state_updated",
            "previous_state": "talking",
            "new_state": "listening",
        }
    )
    assert session._avatar_speaking is False
    assert buf.started == 1
    assert buf.finished == [(1.25, False)]


def test_buffer_acks_do_not_drive_speaking_state(avatar):
    session, buf = avatar
    session._handle_server_event({"type": "agent.audio_buffer_appended"})
    session._handle_server_event({"type": "agent.audio_buffer_committed"})
    assert session._avatar_speaking is False
    assert buf.started == 0
    assert buf.finished == []


def test_speak_started_and_talking_are_idempotent(avatar):
    session, buf = avatar
    session._handle_server_event({"type": "agent.speak_started"})
    session._handle_server_event(
        {"type": "agent.state_updated", "previous_state": "idle", "new_state": "talking"}
    )
    assert session._avatar_speaking is True
    assert buf.started == 1


def test_audio_buffer_cleared_marks_interrupted(avatar):
    session, buf = avatar
    session._handle_server_event(
        {"type": "agent.state_updated", "previous_state": "idle", "new_state": "talking"}
    )
    session._handle_server_event({"type": "agent.audio_buffer_cleared"})
    session._handle_server_event(
        {"type": "agent.state_updated", "previous_state": "talking", "new_state": "idle"}
    )
    assert session._avatar_interrupted is True
    assert session._avatar_speaking is False
    assert buf.finished == []


def test_error_event_is_logged(avatar, caplog: pytest.LogCaptureFixture):
    session, buf = avatar
    with caplog.at_level(logging.ERROR, logger="livekit.plugins.liveavatar"):
        session._handle_server_event(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad audio"},
            }
        )
    assert "invalid_request_error" in caplog.text
    assert session._avatar_speaking is False
    assert buf.started == 0


async def test_barge_in_sends_interrupt_after_talking(avatar):
    session, _buf = avatar
    session._handle_server_event(
        {"type": "agent.state_updated", "previous_state": "idle", "new_state": "talking"}
    )
    session._audio_playing = True
    session._on_clear_buffer()
    if session._tasks:
        await asyncio.gather(*session._tasks)

    msg = session._msg_ch.recv_nowait()
    assert msg["type"] == "agent.interrupt"


async def test_barge_in_skips_interrupt_when_never_talking(avatar):
    session, _buf = avatar
    session._handle_server_event({"type": "agent.audio_buffer_appended"})
    session._audio_playing = True
    session._on_clear_buffer()
    if session._tasks:
        await asyncio.gather(*session._tasks)

    with pytest.raises(ChanEmpty):
        session._msg_ch.recv_nowait()
