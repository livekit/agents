from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.beta.tools.sms import SMSToolConfig, create_sms_tool
from livekit.agents.beta.tools.sms.config import SMSResponse
from livekit.agents.llm import FunctionCall
from livekit.agents.llm.tool_context import ToolError
from livekit.agents.voice import RunContext


def create_mock_context() -> RunContext:
    mock_session = MagicMock()
    mock_session.generate_reply = AsyncMock()
    mock_session.room = MagicMock(remote_participants={})

    mock_speech_handle = MagicMock()
    mock_speech_handle.disallow_interruptions = MagicMock()
    mock_function_call = MagicMock(spec=FunctionCall)

    context = RunContext(
        session=mock_session,
        speech_handle=mock_speech_handle,
        function_call=mock_function_call,
    )
    context.disallow_interruptions = MagicMock()
    return context


def test_sms_provider_detection():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="No SMS provider credentials"):
            create_sms_tool()


async def test_send_sms_success():
    env_vars = {
        "TWILIO_ACCOUNT_SID": "test_sid",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_PHONE_NUMBER": "+11234567890",
    }
    mock_context = create_mock_context()

    with patch.dict("os.environ", env_vars):
        tool = create_sms_tool(SMSToolConfig(to_number="+10987654321"))

    with patch(
        "livekit.agents.beta.tools.sms.send_sms.run_sms_request",
        return_value=SMSResponse(success=True, message_id="SM123", to="+10987654321"),
    ):
        result = await tool({"message": "Test"}, mock_context)

    assert "success" in result


async def test_send_sms_validation_errors():
    env_vars = {
        "TWILIO_ACCOUNT_SID": "test_sid",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_PHONE_NUMBER": "+11234567890",
    }
    mock_context = create_mock_context()

    with patch.dict("os.environ", env_vars):
        tool = create_sms_tool(SMSToolConfig(to_number="+10987654321"))

        with pytest.raises(ToolError, match="Message text is required"):
            await tool({}, mock_context)

        tool = create_sms_tool(SMSToolConfig(to_number="invalid"))
        with pytest.raises(ToolError, match="Invalid phone number format"):
            await tool({"message": "Test"}, mock_context)
