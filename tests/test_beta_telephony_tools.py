from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from livekit import rtc
from livekit.agents.beta.tools import (
    DTMF_CODE_MAP,
    CallTransferTool,
    DTMFKeypadTool,
    ai_callable,
    collect_sip_dtmf,
    on_sip_dtmf,
    telephony,
    validate_dtmf_digits,
    validate_transfer_destination,
)
from livekit.agents.beta.tools.dtmf_tool import DTMFKeypadTool as DTMFKeypadToolFromDTMF
from livekit.agents.beta.tools.transfer_tool import (
    CallTransferTool as CallTransferToolFromTransfer,
)
from livekit.agents.job import JobContext
from livekit.agents.llm import (
    FunctionCall,
    FunctionTool,
    ToolError,
    ToolFlag,
    ToolResult,
    Toolset,
    function_tool,
)
from livekit.agents.llm.utils import function_arguments_to_pydantic_model
from livekit.agents.voice.events import RunContext

pytestmark = [pytest.mark.unit]


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


class _MockRoom:
    """A minimal mock room supporting event subscription and emission."""

    def __init__(
        self,
        local_participant: Any = None,
        remote_participants: dict[str, rtc.RemoteParticipant] | None = None,
    ) -> None:
        self.name = "test_room"
        self.local_participant: Any = None
        if local_participant is None:
            self.local_participant = MagicMock()
            self.local_participant.publish_dtmf = AsyncMock()
            self.local_participant.publish_data = AsyncMock()
        else:
            self.local_participant = local_participant
            if not isinstance(getattr(self.local_participant, "publish_dtmf", None), AsyncMock):
                self.local_participant.publish_dtmf = AsyncMock()
            if not isinstance(getattr(self.local_participant, "publish_data", None), AsyncMock):
                self.local_participant.publish_data = AsyncMock()
        self.remote_participants = remote_participants if remote_participants is not None else {}
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def on(self, event: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        self._handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        if event in self._handlers and callback in self._handlers[event]:
            self._handlers[event].remove(callback)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            handler(*args, **kwargs)


def _make_mock_run_context(
    *,
    linked_participant: rtc.RemoteParticipant | None = None,
    remote_participants: dict[str, rtc.RemoteParticipant] | None = None,
    local_participant: Any = None,
    room: _MockRoom | None = None,
) -> tuple[RunContext[Any], _MockRoom, Any]:
    """Construct a minimal RunContext with mocked session, room_io, and room."""
    if local_participant is None:
        mock_local: Any = MagicMock()
        mock_local.publish_dtmf = AsyncMock()
        mock_local.publish_data = AsyncMock()
    else:
        mock_local = local_participant
        if not isinstance(getattr(mock_local, "publish_dtmf", None), AsyncMock):
            mock_local.publish_dtmf = AsyncMock()
        if not isinstance(getattr(mock_local, "publish_data", None), AsyncMock):
            mock_local.publish_data = AsyncMock()

    mock_room = room or _MockRoom(
        local_participant=mock_local,
        remote_participants=remote_participants,
    )

    mock_room_io = MagicMock(name="mock_room_io")
    mock_room_io.room = mock_room
    mock_room_io.linked_participant = linked_participant

    mock_session = MagicMock(name="mock_session")
    mock_session.room_io = mock_room_io

    speech_handle = MagicMock()
    speech_handle.num_steps = 1
    speech_handle.allow_interruptions = True

    fnc_call = FunctionCall(
        call_id="call_test_123",
        name="test_tool",
        arguments="{}",
        extra={},
    )

    ctx: RunContext[Any] = RunContext(
        session=mock_session,
        speech_handle=speech_handle,
        function_call=fnc_call,
    )
    return ctx, mock_room, mock_local


# ==============================================================================
# Suite 1: Tool Registration, Backward Compatibility & Schema Generation
# ==============================================================================


class TestTelephonyToolRegistrationAndSchema:
    def test_imports_and_aliases(self) -> None:
        """Verify module exports and backward-compatible convenience aliases."""
        assert DTMFKeypadTool is DTMFKeypadToolFromDTMF
        assert DTMFKeypadTool is telephony.DTMFKeypadTool
        assert CallTransferTool is CallTransferToolFromTransfer
        assert CallTransferTool is telephony.CallTransferTool

        assert issubclass(DTMFKeypadTool, Toolset)
        assert issubclass(CallTransferTool, Toolset)

        assert hasattr(telephony, "ai_callable")
        assert ai_callable is function_tool
        assert telephony.ai_callable is function_tool

    def test_dtmf_tool_schema_generation(self) -> None:
        """Verify Pydantic argument schema generation for DTMFKeypadTool."""
        toolset = DTMFKeypadTool()
        assert len(toolset.tools) == 1
        fn_tool = toolset.tools[0]
        assert isinstance(fn_tool, FunctionTool)
        assert fn_tool.info.name == "send_dtmf"

        model: type[BaseModel] = function_arguments_to_pydantic_model(fn_tool)
        schema = model.model_json_schema()
        assert schema["type"] == "object"
        assert "digits" in schema["properties"]
        assert schema["required"] == ["digits"]
        assert "self" not in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_call_transfer_tool_schema_generation(self) -> None:
        """Verify Pydantic argument schema generation for CallTransferTool."""
        toolset = CallTransferTool()
        assert len(toolset.tools) == 1
        fn_tool = toolset.tools[0]
        assert isinstance(fn_tool, FunctionTool)
        assert fn_tool.info.name == "transfer_call"

        model: type[BaseModel] = function_arguments_to_pydantic_model(fn_tool)
        schema = model.model_json_schema()
        assert schema["type"] == "object"
        assert "transfer_to" in schema["properties"]
        assert schema["required"] == ["transfer_to"]
        assert "self" not in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_tool_flags_ignore_on_enter(self) -> None:
        """Verify ignore_on_enter flag propagation."""
        transfer_default = CallTransferTool(ignore_on_enter=False)
        t0 = transfer_default.tools[0]
        assert isinstance(t0, FunctionTool)
        assert not (t0.info.flags & ToolFlag.IGNORE_ON_ENTER)

        transfer_ignored = CallTransferTool(ignore_on_enter=True)
        t1 = transfer_ignored.tools[0]
        assert isinstance(t1, FunctionTool)
        assert t1.info.flags & ToolFlag.IGNORE_ON_ENTER

        dtmf_default = DTMFKeypadTool(ignore_on_enter=False)
        t2 = dtmf_default.tools[0]
        assert isinstance(t2, FunctionTool)
        assert not (t2.info.flags & ToolFlag.IGNORE_ON_ENTER)

        dtmf_ignored = DTMFKeypadTool(ignore_on_enter=True)
        t3 = dtmf_ignored.tools[0]
        assert isinstance(t3, FunctionTool)
        assert t3.info.flags & ToolFlag.IGNORE_ON_ENTER


# ==============================================================================
# Suite 2: DTMF Digits Validation
# ==============================================================================


class TestDTMFDigitsValidation:
    def test_valid_dtmf_digits_pass(self) -> None:
        """Verify valid standard and extended RFC 4733 digits succeed without error."""
        valid_samples = [
            "0",
            "1",
            "9",
            "*",
            "#",
            "A",
            "B",
            "C",
            "D",
            "0123456789*#",
            "ABCD",
            "1234#*ABCD",
        ]
        for sample in valid_samples:
            validate_dtmf_digits(sample)

    def test_invalid_dtmf_digits_rejected(self) -> None:
        """Verify non-DTMF characters, lowercase letters, whitespace, and empty strings raise ToolError."""
        invalid_samples = [
            "",  # empty
            "   ",  # whitespace only
            "1 2",  # embedded whitespace
            " 1",  # leading whitespace
            "1 ",  # trailing whitespace
            "+1555",  # plus symbol
            "$100",  # dollar symbol
            "abcd",  # lowercase letters
            "12X3",  # invalid letter
            "1,2",  # comma
            "1.2",  # dot
            "123-456",  # hyphen
        ]
        for invalid in invalid_samples:
            with pytest.raises(ToolError) as exc_info:
                validate_dtmf_digits(invalid)
            assert "DTMF" in str(exc_info.value)

    def test_dtmf_code_map_completeness(self) -> None:
        """Verify DTMF_CODE_MAP maps all 16 RFC 4733 digits correctly."""
        for i in range(10):
            assert DTMF_CODE_MAP[str(i)] == i
        assert DTMF_CODE_MAP["*"] == 10
        assert DTMF_CODE_MAP["#"] == 11
        assert DTMF_CODE_MAP["A"] == 12
        assert DTMF_CODE_MAP["B"] == 13
        assert DTMF_CODE_MAP["C"] == 14
        assert DTMF_CODE_MAP["D"] == 15


# ==============================================================================
# Suite 3: DTMF Tone Emission & Data Packets
# ==============================================================================


class TestDTMFEmission:
    @pytest.mark.asyncio
    async def test_dtmf_standard_digits_emission(self) -> None:
        """Verify sequential emission of standard DTMF digits with correct codes."""
        ctx, _, mock_local = _make_mock_run_context()
        toolset = DTMFKeypadTool(publish_delay=0.0, enable_data_packets=False)

        res = await toolset.send_dtmf(ctx, "0123456789*#")
        assert isinstance(res, ToolResult)
        assert res.reply_required is False
        assert "Successfully sent DTMF digits: 0123456789*#" in str(res.output)

        assert mock_local.publish_dtmf.await_count == 12
        expected_calls = [
            (0, "0"),
            (1, "1"),
            (2, "2"),
            (3, "3"),
            (4, "4"),
            (5, "5"),
            (6, "6"),
            (7, "7"),
            (8, "8"),
            (9, "9"),
            (10, "*"),
            (11, "#"),
        ]
        for idx, (expected_code, expected_digit) in enumerate(expected_calls):
            call_kwargs = mock_local.publish_dtmf.await_args_list[idx].kwargs
            assert call_kwargs == {"code": expected_code, "digit": expected_digit}

    @pytest.mark.asyncio
    async def test_dtmf_extended_digits_emission(self) -> None:
        """Verify extended RFC 4733 digits A-D map to codes 12-15."""
        ctx, _, mock_local = _make_mock_run_context()
        toolset = DTMFKeypadTool(publish_delay=0.0, enable_data_packets=False)

        res = await toolset.send_dtmf(ctx, "ABCD")
        assert res.reply_required is False
        assert mock_local.publish_dtmf.await_count == 4
        expected_calls = [(12, "A"), (13, "B"), (14, "C"), (15, "D")]
        for idx, (expected_code, expected_digit) in enumerate(expected_calls):
            assert mock_local.publish_dtmf.await_args_list[idx].kwargs == {
                "code": expected_code,
                "digit": expected_digit,
            }

    @pytest.mark.asyncio
    async def test_dtmf_publish_delay_applied(self) -> None:
        """Verify configurable publish_delay sleeps only between consecutive digits."""
        ctx, _, _ = _make_mock_run_context()
        toolset = DTMFKeypadTool(publish_delay=0.15, enable_data_packets=False)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await toolset.send_dtmf(ctx, "123")
            assert mock_sleep.await_count == 2
            mock_sleep.assert_awaited_with(0.15)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await toolset.send_dtmf(ctx, "1")
            mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dtmf_concurrent_calls_do_not_interleave(self) -> None:
        """Verify parallel send_dtmf calls serialize cleanly and preserve inter-sequence pacing."""
        ctx, _, mock_local = _make_mock_run_context()
        published: list[str] = []

        async def _record_publish(*, code: int, digit: str) -> None:
            published.append(digit)
            await asyncio.sleep(0)

        mock_local.publish_dtmf = AsyncMock(side_effect=_record_publish)
        toolset = DTMFKeypadTool(publish_delay=0.01, enable_data_packets=False)

        res1, res2 = await asyncio.gather(
            toolset.send_dtmf(ctx, "12"),
            toolset.send_dtmf(ctx, "34"),
        )
        assert "12" in str(res1.output)
        assert "34" in str(res2.output)
        assert published == ["1", "2", "3", "4"]

    @pytest.mark.asyncio
    async def test_dtmf_data_packet_publishing_enabled(self) -> None:
        """Verify WebRTC data packet publishing with correct JSON payload and topic."""
        ctx, _, mock_local = _make_mock_run_context()
        toolset = DTMFKeypadTool(
            publish_delay=0.0,
            enable_data_packets=True,
            data_packet_topic="custom_dtmf_topic",
        )

        await toolset.send_dtmf(ctx, "1*")
        assert mock_local.publish_data.await_count == 2

        first_call = mock_local.publish_data.await_args_list[0]
        first_payload = json.loads(first_call.args[0].decode("utf-8"))
        assert first_payload == {"event": "dtmf", "digit": "1", "code": 1}
        assert first_call.kwargs["topic"] == "custom_dtmf_topic"
        assert first_call.kwargs["reliable"] is True

        second_call = mock_local.publish_data.await_args_list[1]
        second_payload = json.loads(second_call.args[0].decode("utf-8"))
        assert second_payload == {"event": "dtmf", "digit": "*", "code": 10}

    @pytest.mark.asyncio
    async def test_dtmf_data_packet_publishing_disabled(self) -> None:
        """Verify no WebRTC data packets published when enable_data_packets=False."""
        ctx, _, mock_local = _make_mock_run_context()
        toolset = DTMFKeypadTool(publish_delay=0.0, enable_data_packets=False)

        await toolset.send_dtmf(ctx, "5")
        mock_local.publish_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_dtmf_publish_error_surfaced_as_tool_error(self) -> None:
        """Verify RTC publish failures raise ToolError with descriptive diagnostic."""
        ctx, _, mock_local = _make_mock_run_context()
        mock_local.publish_dtmf = AsyncMock(side_effect=RuntimeError("WebRTC channel dropped"))
        toolset = DTMFKeypadTool(publish_delay=0.0)

        with pytest.raises(ToolError) as exc_info:
            await toolset.send_dtmf(ctx, "1")
        assert "Failed to publish DTMF digit '1'" in str(exc_info.value)
        assert "WebRTC channel dropped" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_dtmf_no_local_participant_raises_tool_error(self) -> None:
        """Verify ToolError is raised when room has no local participant."""
        ctx, mock_room, _ = _make_mock_run_context()
        mock_room.local_participant = None
        toolset = DTMFKeypadTool(publish_delay=0.0)

        with pytest.raises(ToolError) as exc_info:
            await toolset.send_dtmf(ctx, "1")
        assert "no local participant" in str(exc_info.value).lower()


# ==============================================================================
# Suite 4: Inbound SIP DTMF Interception & Helpers
# ==============================================================================


class TestInboundSIPDTMFInterception:
    def test_keypad_listen_and_stop_listening(self) -> None:
        """Verify DTMFKeypadTool listen and stop_listening register/unregister event hooks."""
        room = _MockRoom()
        received: list[rtc.SipDTMF] = []

        keypad = DTMFKeypadTool(on_dtmf_received=lambda ev: received.append(ev))
        keypad.listen(room)  # type: ignore[arg-type]

        # Simulate inbound SIP DTMF event
        dummy_ev = rtc.SipDTMF(code=1, digit="1", participant=None)
        room.emit("sip_dtmf_received", dummy_ev)
        assert len(received) == 1
        assert received[0].digit == "1"

        # Stop listening and verify no more events received
        keypad.stop_listening()
        room.emit("sip_dtmf_received", rtc.SipDTMF(code=2, digit="2", participant=None))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_keypad_async_callback_and_aclose_cleanup(self) -> None:
        """Verify DTMFKeypadTool handles async callbacks and cleans up in aclose."""
        room = _MockRoom()
        received: list[str] = []

        async def _async_cb(ev: rtc.SipDTMF) -> None:
            await asyncio.sleep(0.01)
            received.append(ev.digit)

        keypad = DTMFKeypadTool(on_dtmf_received=_async_cb)
        keypad.listen(room)  # type: ignore[arg-type]

        room.emit("sip_dtmf_received", rtc.SipDTMF(code=9, digit="9", participant=None))
        await asyncio.sleep(0.05)
        assert received == ["9"]

        await keypad.aclose()
        assert keypad._subscribed_room is None

    @pytest.mark.asyncio
    async def test_keypad_setup_subscribes_via_job_context_when_no_explicit_room(self) -> None:
        """Verify DTMFKeypadTool.setup() subscribes to job_ctx.room when no explicit room was given."""
        room = _MockRoom()
        received: list[str] = []
        mock_job_ctx = MagicMock(spec=JobContext, room=room)

        keypad = DTMFKeypadTool(on_dtmf_received=lambda ev: received.append(ev.digit))
        assert keypad._subscribed_room is None

        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            await keypad.setup()

        assert keypad._subscribed_room is room
        room.emit("sip_dtmf_received", rtc.SipDTMF(code=5, digit="5", participant=None))
        assert received == ["5"]

        await keypad.aclose()
        assert keypad._subscribed_room is None
        room.emit("sip_dtmf_received", rtc.SipDTMF(code=6, digit="6", participant=None))
        assert received == ["5"]

    @pytest.mark.asyncio
    async def test_send_dtmf_subscribes_fallback_when_room_resolved(self) -> None:
        """Verify send_dtmf registers on_dtmf_received if setup() ran before room was available."""
        received: list[str] = []
        ctx, mock_room, _ = _make_mock_run_context()
        # Use _MockRoom event hooks on mock_room
        backing_room = _MockRoom()
        mock_room.on = backing_room.on
        mock_room.off = backing_room.off

        keypad = DTMFKeypadTool(
            publish_delay=0.0,
            enable_data_packets=False,
            on_dtmf_received=lambda ev: received.append(ev.digit),
        )
        assert keypad._subscribed_room is None

        await keypad.send_dtmf(ctx, "1")
        assert keypad._subscribed_room is mock_room
        backing_room.emit("sip_dtmf_received", rtc.SipDTMF(code=7, digit="7", participant=None))
        assert received == ["7"]
        await keypad.aclose()

    @pytest.mark.asyncio
    async def test_send_dtmf_does_not_restart_stopped_listener(self) -> None:
        """Verify send_dtmf does not reattach listener after explicit stop_listening/unregister_listener."""
        received: list[str] = []
        ctx, mock_room, _ = _make_mock_run_context()
        backing_room = _MockRoom()
        mock_room.on = backing_room.on
        mock_room.off = backing_room.off

        keypad = DTMFKeypadTool(
            publish_delay=0.0,
            enable_data_packets=False,
            room=mock_room,  # type: ignore[arg-type]
            on_dtmf_received=lambda ev: received.append(ev.digit),
        )
        assert keypad._subscribed_room is mock_room

        keypad.stop_listening()
        assert keypad._subscribed_room is None

        await keypad.send_dtmf(ctx, "1")
        assert keypad._subscribed_room is None
        backing_room.emit("sip_dtmf_received", rtc.SipDTMF(code=8, digit="8", participant=None))
        assert received == []

    @pytest.mark.asyncio
    async def test_keypad_setup_aclose_setup_resubscribes_after_handoff(self) -> None:
        """Verify setup() -> aclose() -> setup() re-subscribes on the same keypad after agent handoff."""
        room = _MockRoom()
        received: list[str] = []
        mock_job_ctx = MagicMock(spec=JobContext, room=room)

        keypad = DTMFKeypadTool(on_dtmf_received=lambda ev: received.append(ev.digit))

        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            await keypad.setup()
            assert keypad._subscribed_room is room
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=1, digit="1", participant=None))
            assert received == ["1"]

            # Simulate AgentActivity.aclose() on handoff
            await keypad.aclose()
            assert keypad._subscribed_room is None
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=2, digit="2", participant=None))
            assert received == ["1"]

            # Simulate AgentActivity._setup_toolsets() when returning to the agent
            await keypad.setup()
            assert keypad._subscribed_room is room
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=3, digit="3", participant=None))
            assert received == ["1", "3"]

        await keypad.aclose()

    def test_on_sip_dtmf_helper_filtering(self) -> None:
        """Verify on_sip_dtmf helper filters by participant and unregisters cleanly."""
        room = _MockRoom()
        target_p = MagicMock(spec=rtc.RemoteParticipant, identity="caller_101")
        other_p = MagicMock(spec=rtc.RemoteParticipant, identity="caller_999")

        collected: list[str] = []
        unsubscribe = on_sip_dtmf(
            room,  # type: ignore[arg-type]
            lambda ev: collected.append(ev.digit),
            participant="caller_101",
        )

        # Event from target participant should be processed
        room.emit(
            "sip_dtmf_received",
            rtc.SipDTMF(code=1, digit="1", participant=target_p),
        )
        assert collected == ["1"]

        # Event from other participant should be filtered out
        room.emit(
            "sip_dtmf_received",
            rtc.SipDTMF(code=2, digit="2", participant=other_p),
        )
        assert collected == ["1"]

        # After unsubscribe, events are ignored
        unsubscribe()
        room.emit(
            "sip_dtmf_received",
            rtc.SipDTMF(code=3, digit="3", participant=target_p),
        )
        assert collected == ["1"]

    @pytest.mark.asyncio
    async def test_collect_sip_dtmf_helper_by_num_digits(self) -> None:
        """Verify collect_sip_dtmf collects fixed number of digits and returns string."""
        room = _MockRoom()

        async def _emit_delayed_digits() -> None:
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=1, digit="1", participant=None))
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=2, digit="2", participant=None))
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=3, digit="3", participant=None))

        emit_task = asyncio.create_task(_emit_delayed_digits())
        result = await collect_sip_dtmf(room, num_digits=3, timeout=2.0)  # type: ignore[arg-type]
        await emit_task
        assert result == "123"

    @pytest.mark.asyncio
    async def test_collect_sip_dtmf_helper_by_stop_digit(self) -> None:
        """Verify collect_sip_dtmf terminates collection upon receiving stop_digit."""
        room = _MockRoom()

        async def _emit_delayed_digits() -> None:
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=4, digit="4", participant=None))
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=5, digit="5", participant=None))
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=11, digit="#", participant=None))
            await asyncio.sleep(0.02)
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=6, digit="6", participant=None))

        emit_task = asyncio.create_task(_emit_delayed_digits())
        result = await collect_sip_dtmf(room, stop_digit="#", timeout=2.0)  # type: ignore[arg-type]
        await emit_task
        assert result == "45"

    @pytest.mark.asyncio
    async def test_collect_sip_dtmf_helper_timeout(self) -> None:
        """Verify collect_sip_dtmf raises asyncio.TimeoutError when timeout expires."""
        room = _MockRoom()
        with pytest.raises(asyncio.TimeoutError):
            await collect_sip_dtmf(room, num_digits=4, timeout=0.05)  # type: ignore[arg-type]


# ==============================================================================
# Suite 5: Call Transfer Destination Validation
# ==============================================================================


class TestCallTransferDestinationValidation:
    def test_valid_e164_destinations(self) -> None:
        """Verify standard, formatted, and tel: E.164 phone numbers normalize to tel: URIs."""
        assert validate_transfer_destination("+15551234567") == "tel:+15551234567"
        assert validate_transfer_destination("+442071838750") == "tel:+442071838750"
        assert validate_transfer_destination("tel:+15551234567") == "tel:+15551234567"
        assert validate_transfer_destination("TEL:+1 (555) 123-4567") == "tel:+15551234567"
        # Formatted with spaces, dashes, parentheses
        assert validate_transfer_destination("+1 (555) 123-4567") == "tel:+15551234567"
        assert validate_transfer_destination("+1-555-123-4567") == "tel:+15551234567"
        assert validate_transfer_destination("+1.555.123.4567") == "tel:+15551234567"

    def test_valid_pbx_extensions(self) -> None:
        """Verify internal PBX extensions require sip_domain or resolve to sip:ext@domain."""
        assert validate_transfer_destination("101", sip_domain="pbx.example.com") == (
            "sip:101@pbx.example.com"
        )
        assert validate_transfer_destination("1001", sip_domain="pbx.example.com") == (
            "sip:1001@pbx.example.com"
        )
        with pytest.raises(ToolError, match="sip_domain"):
            validate_transfer_destination("1001")

    def test_valid_sip_uris(self) -> None:
        """Verify SIP and SIPS URIs normalize scheme to lowercase."""
        assert (
            validate_transfer_destination("sip:agent@trunk.example.com")
            == "sip:agent@trunk.example.com"
        )
        assert (
            validate_transfer_destination("sips:secure@pbx.internal") == "sips:secure@pbx.internal"
        )
        assert (
            validate_transfer_destination("SIP:dispatcher@192.168.1.50")
            == "sip:dispatcher@192.168.1.50"
        )

    def test_invalid_transfer_destinations_raise_tool_error(self) -> None:
        """Verify invalid destinations raise ToolError."""
        invalid_destinations = [
            "",  # empty
            "   ",  # whitespace
            "12",  # 2 digits (too short for PBX, not E.164)
            "12345678",  # 8 digits without + prefix (too long for PBX)
            "invalid_text",  # non-numeric text
            "http://example.com",  # HTTP URI instead of SIP
            "sip:invalid",  # SIP without @
            "sips:nohost",  # SIPS without @
            "+",  # lone plus
            "+0123456",  # country code cannot start with 0
        ]
        for dest in invalid_destinations:
            with pytest.raises(ToolError) as exc_info:
                validate_transfer_destination(dest)
            assert "Invalid" in str(exc_info.value) or "empty" in str(exc_info.value).lower()


# ==============================================================================
# Suite 6: Call Transfer Execution & Resolution
# ==============================================================================


class TestCallTransferExecution:
    @pytest.mark.asyncio
    async def test_call_transfer_linked_sip_participant_success(self) -> None:
        """Verify transfer executes using linked_participant when it is a SIP participant."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_caller_linked",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)

        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock()

        toolset = CallTransferTool(play_dialtone=False)
        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            res = await toolset.transfer_call(ctx, "+1 (555) 123-4567")

        assert isinstance(res, ToolResult)
        assert res.reply_required is False
        assert "tel:+15551234567" in str(res.output)

        mock_job_ctx.transfer_sip_participant.assert_awaited_once_with(
            sip_p,
            transfer_to="tel:+15551234567",
            play_dialtone=False,
        )

    @pytest.mark.asyncio
    async def test_call_transfer_fallback_to_room_remote_participants(self) -> None:
        """Verify discovery of SIP participant from room remote_participants when linked is None."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_caller_room",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        std_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="web_user",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        )

        ctx, _, _ = _make_mock_run_context(
            linked_participant=None,
            remote_participants={"web": std_p, "sip": sip_p},
        )

        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock()

        toolset = CallTransferTool(play_dialtone=False)
        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            res = await toolset.transfer_call(ctx, "sip:escalations@pbx.internal.net")

        assert res.reply_required is False
        assert "sip:escalations@pbx.internal.net" in str(res.output)
        mock_job_ctx.transfer_sip_participant.assert_awaited_once_with(
            sip_p,
            transfer_to="sip:escalations@pbx.internal.net",
            play_dialtone=False,
        )

    @pytest.mark.asyncio
    async def test_call_transfer_play_dialtone_configuration(self) -> None:
        """Verify play_dialtone parameter default and invocation override."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_p",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)

        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock()

        # Tool configured with play_dialtone=True and sip_domain
        toolset = CallTransferTool(play_dialtone=True, sip_domain="pbx.example.com")
        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            await toolset.transfer_call(ctx, "1001")
            mock_job_ctx.transfer_sip_participant.assert_awaited_with(
                sip_p,
                transfer_to="sip:1001@pbx.example.com",
                play_dialtone=True,
            )

            # Override via method argument
            await toolset.transfer_call(ctx, "1001", play_dialtone=False)
            mock_job_ctx.transfer_sip_participant.assert_awaited_with(
                sip_p,
                transfer_to="sip:1001@pbx.example.com",
                play_dialtone=False,
            )


# ==============================================================================
# Suite 7: Call Transfer Error Handling
# ==============================================================================


class TestCallTransferErrorHandling:
    @pytest.mark.asyncio
    async def test_error_when_no_sip_participant_found(self) -> None:
        """Verify ToolError when room has only standard WebRTC participants."""
        std_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="web_user",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        )
        ctx, _, _ = _make_mock_run_context(
            linked_participant=std_p,
            remote_participants={"web": std_p},
        )

        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock()

        toolset = CallTransferTool()
        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            with pytest.raises(ToolError) as exc_info:
                await toolset.transfer_call(ctx, "+15551234567")

        assert "No active SIP participant found to transfer." in str(exc_info.value)
        mock_job_ctx.transfer_sip_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_when_invalid_destination_passed(self) -> None:
        """Verify ToolError on invalid destination and transfer API is not invoked."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_p",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)

        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock()

        toolset = CallTransferTool()
        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            with pytest.raises(ToolError) as exc_info:
                await toolset.transfer_call(ctx, "invalid_target")

        assert "Invalid" in str(exc_info.value)
        mock_job_ctx.transfer_sip_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_when_rpc_call_fails(self) -> None:
        """Verify Twirp / LiveKit server RPC errors are wrapped cleanly in ToolError."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_p",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)

        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock(
            side_effect=RuntimeError("Twirp error: participant not found")
        )

        toolset = CallTransferTool()
        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            with pytest.raises(ToolError) as exc_info:
                await toolset.transfer_call(ctx, "+15551234567")

        assert "Call transfer failed" in str(exc_info.value)
        assert "Twirp error: participant not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_when_job_context_missing(self) -> None:
        """Verify ToolError when JobContext is None."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_p",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)

        toolset = CallTransferTool()
        with patch("livekit.agents.beta.tools.telephony.get_job_context", return_value=None):
            with pytest.raises(ToolError) as exc_info:
                await toolset.transfer_call(ctx, "+15551234567")

        assert "No active JobContext found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_dtmf_does_not_log_sensitive_digits(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify send_dtmf debug logging does not expose DTMF PINs or digits."""
        ctx, _, _ = _make_mock_run_context()
        toolset = DTMFKeypadTool(publish_delay=0.0, enable_data_packets=False)

        with caplog.at_level("DEBUG", logger="livekit.agents"):
            await toolset.send_dtmf(ctx, "4321#")

        assert "4321#" not in caplog.text

    @pytest.mark.asyncio
    async def test_transfer_call_does_not_log_sensitive_destination(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify transfer_call debug logging does not expose transfer_to destination."""
        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_p",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)
        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock(return_value=None)

        toolset = CallTransferTool()
        with (
            patch("livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx),
            caplog.at_level("DEBUG", logger="livekit.agents"),
        ):
            await toolset.transfer_call(ctx, "+15559876543")

        assert "+15559876543" not in caplog.text

    @pytest.mark.asyncio
    async def test_error_when_transfer_response_status_is_failed(self) -> None:
        """Verify ToolError is raised and on_tool_completed is skipped when SIP status is STS_TRANSFER_FAILED."""
        from livekit.protocol import sip

        sip_p = MagicMock(
            spec=rtc.RemoteParticipant,
            identity="sip_p",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        )
        ctx, _, _ = _make_mock_run_context(linked_participant=sip_p)
        mock_job_ctx = MagicMock(spec=JobContext)
        mock_job_ctx.transfer_sip_participant = AsyncMock(
            return_value=sip.TransferSIPParticipantResponse(
                status=sip.SIPTransferStatus.STS_TRANSFER_FAILED,
                reason=sip.SIPTransferReason.STR_CALL_ENDED,
                sip_status=sip.SIPStatus(code=487, status="Request Terminated"),
            )
        )
        on_completed = AsyncMock()
        toolset = CallTransferTool(on_tool_completed=on_completed)

        with patch(
            "livekit.agents.beta.tools.telephony.get_job_context", return_value=mock_job_ctx
        ):
            with pytest.raises(ToolError, match="Call transfer failed"):
                await toolset.transfer_call(ctx, "+15551234567")

        on_completed.assert_not_awaited()
