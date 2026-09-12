from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.beta.workflows.warm_transfer import (
    WarmTransferError,
    WarmTransferFailure,
    WarmTransferTask,
)
from livekit.agents.llm.tool_context import ToolError

pytestmark = pytest.mark.unit


def test_warm_transfer_error_inheritance_and_properties() -> None:
    err = WarmTransferError(
        "human agent declined to connect: busy",
        code=WarmTransferFailure.DECLINED,
        reason="busy",
    )
    assert isinstance(err, ToolError)
    assert err.code == WarmTransferFailure.DECLINED
    assert err.reason == "busy"
    assert str(err) == "human agent declined to connect: busy"


@pytest.mark.asyncio
async def test_decline_transfer_creates_structured_error() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()

    await task.decline_transfer("agent is in a meeting")

    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.DECLINED
    assert result.reason == "agent is in a meeting"
    assert "human agent declined to connect: agent is in a meeting" in str(result)


@pytest.mark.asyncio
async def test_voicemail_detected_creates_structured_error() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()

    await task.voicemail_detected()

    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.VOICEMAIL
    assert str(result) == "voicemail detected"


def test_human_agent_room_close_with_destination_left() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_event_loop().create_future()

    # Pre-recorded destination departure
    task._destination_disconnect_reason = rtc.DisconnectReason.USER_UNAVAILABLE
    task._destination_call_status = "busy"

    task._on_human_agent_room_close(rtc.DisconnectReason.ROOM_DELETED)

    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.DESTINATION_LEFT
    assert result.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE
    assert result.call_status == "busy"
    assert "destination left: USER_UNAVAILABLE" in str(result)


def test_human_agent_room_close_without_destination_left() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_event_loop().create_future()

    task._destination_disconnect_reason = None
    task._destination_call_status = None

    task._on_human_agent_room_close(rtc.DisconnectReason.SERVER_SHUTDOWN)

    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.ROOM_CLOSED
    assert result.disconnect_reason == rtc.DisconnectReason.SERVER_SHUTDOWN
    assert "room closed: SERVER_SHUTDOWN" in str(result)


@pytest.mark.asyncio
async def test_twilio_connector_warm_transfer_initializes_destination_state() -> None:
    from livekit.agents.beta.workflows.warm_transfer import TwilioConnectorWarmTransferTask

    task = TwilioConnectorWarmTransferTask(
        phone_number="+1234567890",
        twilio_from_number="+1098765432",
        twilio_account_sid="AC123",
        twilio_auth_token="secret",
    )
    try:
        assert hasattr(task, "_destination_disconnect_reason")
        assert task._destination_disconnect_reason is None
        assert hasattr(task, "_destination_call_status")
        assert task._destination_call_status is None
        assert hasattr(task, "_human_agent_participant_disconnected_cb")
        assert task._human_agent_participant_disconnected_cb is None
    finally:
        await task._background_audio._audio_mixer.aclose()
