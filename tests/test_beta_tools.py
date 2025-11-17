from __future__ import annotations

from unittest.mock import patch

import pytest

from livekit.agents.beta.tools.sms import (
    SMSToolConfig,
    create_sms_tool,
)
from livekit.agents.llm.tool_context import ToolError


def test_sms_provider_detection():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="No SMS provider credentials"):
            create_sms_tool()


class MockSMSResponse:
    def __init__(self, status: int = 200, json_data: dict | None = None):
        self.status = status
        self._json_data = json_data or {}

    async def json(self):
        return self._json_data

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSMSSession:
    def __init__(self, response: MockSMSResponse | None = None):
        self.response = response or MockSMSResponse()

    def post(self, url: str, **kwargs):
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


async def test_send_sms_success():
    env_vars = {
        "TWILIO_ACCOUNT_SID": "test_sid",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_PHONE_NUMBER": "+11234567890",
    }
    config = SMSToolConfig(to_number="+10987654321")
    mock_response = MockSMSResponse(status=200, json_data={"sid": "SM123"})
    mock_session = MockSMSSession(response=mock_response)

    with patch.dict("os.environ", env_vars):
        tool = create_sms_tool(config)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await tool({"message": "Test"})
        assert "success" in result


async def test_send_sms_validation_errors():
    env_vars = {
        "TWILIO_ACCOUNT_SID": "test_sid",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_PHONE_NUMBER": "+11234567890",
    }

    with patch.dict("os.environ", env_vars):
        tool = create_sms_tool(SMSToolConfig(to_number="+10987654321"))

        with pytest.raises(ToolError, match="Message text is required"):
            await tool({})

        tool = create_sms_tool(SMSToolConfig(to_number="invalid"))
        with pytest.raises(ToolError, match="Invalid phone number format"):
            await tool({"message": "Test"})
