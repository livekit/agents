from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
async def test_human_agent_room_close_with_destination_left() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()

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


@pytest.mark.asyncio
async def test_human_agent_room_close_without_destination_left() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()

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
async def test_human_agent_participant_disconnected_completes_transfer() -> None:
    task = object.__new__(WarmTransferTask)
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()
    task._human_agent_identity = "human-agent-sip"

    # Other participant disconnecting should be ignored
    other_participant = MagicMock(spec=rtc.RemoteParticipant)
    other_participant.identity = "other-participant"
    task._on_human_agent_participant_disconnected(other_participant)
    task.complete.assert_not_called()
    assert not task._human_agent_failed_fut.done()

    # Destination participant disconnecting with USER_UNAVAILABLE completes transfer
    dest_participant = MagicMock(spec=rtc.RemoteParticipant)
    dest_participant.identity = "human-agent-sip"
    dest_participant.disconnect_reason = rtc.DisconnectReason.USER_UNAVAILABLE
    dest_participant.attributes = {"sip.callStatus": "busy"}

    task._on_human_agent_participant_disconnected(dest_participant)

    task.complete.assert_called_once()
    assert task._human_agent_failed_fut.done()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.DESTINATION_LEFT
    assert result.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE
    assert result.call_status == "busy"
    assert "destination left: USER_UNAVAILABLE" in str(result)


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
        assert task._human_agent_participant_disconnected_cb is not None
        assert callable(task._human_agent_participant_disconnected_cb)
    finally:
        await task._background_audio._audio_mixer.aclose()


@pytest.mark.asyncio
async def test_on_enter_dial_failure_chains_original_exception() -> None:
    task = object.__new__(WarmTransferTask)
    task._hold_audio = None
    task._caller_room = MagicMock()
    task._set_io_enabled = MagicMock()
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()

    mock_job_ctx = MagicMock()
    mock_job_ctx.room = MagicMock()

    orig_exc = RuntimeError("SIP gateway error 503")

    async def _failing_dial():
        raise orig_exc

    task._dial_human_agent = _failing_dial

    with patch(
        "livekit.agents.beta.workflows.warm_transfer.get_job_context", return_value=mock_job_ctx
    ):
        await task.on_enter()

    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.DIAL_FAILED
    assert result.__cause__ is orig_exc


@pytest.mark.asyncio
async def test_on_enter_shuts_down_child_session_when_failed_fut_done() -> None:
    task = object.__new__(WarmTransferTask)
    task._hold_audio = None
    task._caller_room = MagicMock()
    task._set_io_enabled = MagicMock()
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()

    mock_job_ctx = MagicMock()
    mock_job_ctx.room = MagicMock()
    mock_child_sess = MagicMock()

    async def _successful_dial():
        # Destination disconnected while dial was completing
        task._human_agent_failed_fut.set_result(None)
        return mock_child_sess

    task._dial_human_agent = _successful_dial

    with patch(
        "livekit.agents.beta.workflows.warm_transfer.get_job_context", return_value=mock_job_ctx
    ):
        await task.on_enter()

    # The session must be shut down and not leaked
    mock_child_sess.shutdown.assert_called_once()
    assert task._human_agent_sess is None


@pytest.mark.asyncio
async def test_on_enter_reraises_dial_error_when_failed_fut_also_done() -> None:
    task = object.__new__(WarmTransferTask)
    task._hold_audio = None
    task._caller_room = MagicMock()
    task._set_io_enabled = MagicMock()
    task._human_agent_sess = None
    task._hold_audio_handle = None
    task.done = MagicMock(return_value=False)
    task.complete = MagicMock()
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()

    mock_job_ctx = MagicMock()
    mock_job_ctx.room = MagicMock()
    sip_exc = RuntimeError("SIP status 503 Service Unavailable")

    async def _failing_dial():
        task._human_agent_failed_fut.set_result(None)
        raise sip_exc

    task._dial_human_agent = _failing_dial

    with patch(
        "livekit.agents.beta.workflows.warm_transfer.get_job_context", return_value=mock_job_ctx
    ):
        await task.on_enter()

    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.DIAL_FAILED
    # Must preserve the real dial error as cause rather than generic RuntimeError
    assert result.__cause__ is sip_exc


@pytest.mark.asyncio
async def test_merge_calls_missed_destination_departure() -> None:
    task = object.__new__(WarmTransferTask)
    task._caller_room = MagicMock()
    task._caller_room.name = "caller-room"
    task._human_agent_identity = "dest-agent"
    task._destination_disconnect_reason = rtc.DisconnectReason.USER_UNAVAILABLE
    task._destination_call_status = "busy"
    task._human_agent_failed_fut = asyncio.get_running_loop().create_future()
    task._human_agent_participant_disconnected_cb = MagicMock()
    task._on_human_agent_room_close = MagicMock()
    task._hold_audio_handle = None
    task._set_io_enabled = MagicMock()
    task.complete = MagicMock()
    task.done = MagicMock(return_value=False)

    mock_human_sess = MagicMock()
    mock_human_room = MagicMock()
    mock_human_room.name = "human-room"
    # Remote participants does NOT contain dest-agent (already departed)
    mock_human_room.remote_participants = {}
    mock_human_sess.room_io.room = mock_human_room
    task._human_agent_sess = mock_human_sess

    mock_job_ctx = MagicMock()
    mock_job_ctx.api.room.move_participant = AsyncMock(
        side_effect=RuntimeError("participant not found")
    )

    with patch(
        "livekit.agents.beta.workflows.warm_transfer.get_job_context", return_value=mock_job_ctx
    ):
        with pytest.raises(RuntimeError, match="participant not found"):
            await task._merge_calls()

    # Even though move_participant failed, task should complete with DESTINATION_LEFT
    assert task._human_agent_failed_fut.done()
    task.complete.assert_called_once()
    result = task.complete.call_args[0][0]
    assert isinstance(result, WarmTransferError)
    assert result.code == WarmTransferFailure.DESTINATION_LEFT
    assert result.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE
    assert result.call_status == "busy"
