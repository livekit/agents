from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.beta.tools.http import (
    HTTPToolConfig,
    create_http_tool,
)
from livekit.agents.beta.tools.sms import (
    SMSToolConfig,
    create_sms_tool,
)
from livekit.agents.llm import FunctionCall
from livekit.agents.llm.tool_context import ToolError
from livekit.agents.voice import RunContext


class MockResponse:
    def __init__(
        self,
        status: int = 200,
        text: str = '{"result": "success"}',
        headers: dict[str, str] | None = None,
    ):
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    def __init__(self, response: MockResponse | None = None):
        self.response = response or MockResponse()
        self.request_kwargs: dict[str, Any] = {}

    def _store_request(self, method: str, **kwargs):
        self.request_kwargs = {"method": method, **kwargs}
        return self.response

    def get(self, url: str, **kwargs):
        return self._store_request("GET", url=url, **kwargs)

    def post(self, url: str, **kwargs):
        return self._store_request("POST", url=url, **kwargs)

    def put(self, url: str, **kwargs):
        return self._store_request("PUT", url=url, **kwargs)

    def patch(self, url: str, **kwargs):
        return self._store_request("PATCH", url=url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self._store_request("DELETE", url=url, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def create_mock_context() -> RunContext:
    """Create a mock RunContext for testing."""
    mock_session = MagicMock()
    mock_speech_handle = MagicMock()
    mock_speech_handle.num_steps = 1
    mock_function_call = MagicMock(spec=FunctionCall)

    context = RunContext(
        session=mock_session,
        speech_handle=mock_speech_handle,
        function_call=mock_function_call,
    )
    return context


def test_create_tool_validation():
    with pytest.raises(ValueError, match="URL is required"):
        create_http_tool(HTTPToolConfig(url=""))

    with pytest.raises(ValueError, match="Unsupported HTTP method"):
        create_http_tool(HTTPToolConfig(url="https://api.example.com", method="TRACE"))


@pytest.mark.parametrize(
    "method",
    ["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def test_http_methods(method: str):
    config = HTTPToolConfig(
        url="https://api.example.com/test",
        method=method,
    )
    tool = create_http_tool(config)

    mock_response = MockResponse(status=200, text="success")
    mock_session = MockSession(response=mock_response)
    mock_context = create_mock_context()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await tool({}, mock_context)

    assert result == "success"
    assert mock_session.request_kwargs["method"] == method


async def test_http_request_types():
    mock_context = create_mock_context()
    mock_session = MockSession()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        # GET sends params
        tool = create_http_tool(HTTPToolConfig(url="https://api.example.com", method="GET"))
        await tool({"query": "test"}, mock_context)
        assert "params" in mock_session.request_kwargs

        # POST sends json
        tool = create_http_tool(HTTPToolConfig(url="https://api.example.com", method="POST"))
        await tool({"name": "test"}, mock_context)
        assert "json" in mock_session.request_kwargs


async def test_http_error_handling():
    mock_context = create_mock_context()
    tool = create_http_tool(HTTPToolConfig(url="https://api.example.com"))

    mock_response = MockResponse(status=404, text="Not found")
    mock_session = MockSession(response=mock_response)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(ToolError, match="HTTP 404"):
            await tool({}, mock_context)


async def test_output_normalizer():
    mock_context = create_mock_context()
    mock_session = MockSession()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        tool = create_http_tool(
            HTTPToolConfig(
                url="https://api.example.com",
                output_normalizer=lambda resp: f"Status: {resp.status_code}",
            )
        )
        result = await tool({}, mock_context)
        assert result == "Status: 200"


# ===========================
# SMS Tool Tests
# ===========================


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
