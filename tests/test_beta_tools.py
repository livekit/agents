from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.beta.tools.http import (
    HTTPToolConfig,
    HTTPToolParam,
    create_http_tool,
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
    mock_speech_handle.disallow_interruptions = MagicMock()
    mock_function_call = MagicMock(spec=FunctionCall)

    context = RunContext(
        session=mock_session,
        speech_handle=mock_speech_handle,
        function_call=mock_function_call,
    )
    context.disallow_interruptions = MagicMock()
    return context


def test_create_tool_validation():
    with pytest.raises(ValueError, match="URL is required"):
        create_http_tool(HTTPToolConfig(name="status", url=""))

    with pytest.raises(ValueError, match="Tool name is required"):
        create_http_tool(HTTPToolConfig(url="https://api.example.com"))

    with pytest.raises(ValueError, match="Unsupported HTTP method"):
        create_http_tool(
            HTTPToolConfig(name="bad_method", url="https://api.example.com", method="TRACE")
        )


def test_json_schema_params_exclusive():
    with pytest.raises(ValueError, match="Cannot set both 'json_schema' and 'params'"):
        create_http_tool(
            HTTPToolConfig(
                url="https://api.example.com/tasks/:id",
                json_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                params=[
                    HTTPToolParam(name="id", description="task id"),
                ],
            )
        )


@pytest.mark.parametrize(
    "method",
    ["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def test_http_methods(method: str):
    config = HTTPToolConfig(
        name=f"tool_{method.lower()}",
        url="https://api.example.com/test",
        method=method,
    )
    tool = create_http_tool(config)

    mock_response = MockResponse(status=200, text="success")
    mock_session = MockSession(response=mock_response)
    mock_context = create_mock_context()

    with patch("livekit.agents.beta.tools.http.send_http.http_session", return_value=mock_session):
        result = await tool({}, mock_context)

    assert result == "success"
    assert mock_session.request_kwargs["method"] == method
    assert mock_context.disallow_interruptions.called == (method != "GET")


async def test_http_error_handling():
    mock_context = create_mock_context()
    tool = create_http_tool(HTTPToolConfig(name="default_tool", url="https://api.example.com"))

    mock_response = MockResponse(status=404, text="Not found")
    mock_session = MockSession(response=mock_response)

    with patch("livekit.agents.beta.tools.http.send_http.http_session", return_value=mock_session):
        with pytest.raises(ToolError, match="HTTP 404"):
            await tool({}, mock_context)


async def test_url_templating():
    """Test URL templating with :param syntax and argument partitioning."""
    mock_context = create_mock_context()
    mock_session = MockSession()

    # Test with URL template and mixed parameters
    tool = create_http_tool(
        HTTPToolConfig(
            name="url_templating_tool",
            url="https://api.example.com/users/:user_id/posts/:post_id",
            method="PUT",
            json_schema={
                "type": "object",
                "properties": {
                    "user_id": "the user id",  # URL param (shorthand)
                    "post_id": {"type": "string", "description": "the post id"},  # URL param (full)
                    "title": {"type": "string", "description": "post title"},  # Body param
                    "content": {"type": "string", "description": "post content"},  # Body param
                },
                "required": ["user_id", "post_id"],
            },
        )
    )

    with patch("livekit.agents.beta.tools.http.send_http.http_session", return_value=mock_session):
        await tool(
            {"user_id": "123", "post_id": "456", "title": "Test", "content": "Content"},
            mock_context,
        )

    assert mock_session.request_kwargs["url"] == "https://api.example.com/users/123/posts/456"
    assert mock_session.request_kwargs["json"] == {"title": "Test", "content": "Content"}


async def test_argument_sanitization_and_validation():
    """Ensure JSON schema filtering and validation works."""
    mock_context = create_mock_context()
    mock_session = MockSession()

    tool = create_http_tool(
        HTTPToolConfig(
            name="sanitize_tool",
            url="https://api.example.com/tasks",
            method="POST",
            json_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "task id"},
                    "title": {"type": "string", "description": "task title"},
                },
                "required": ["id"],
            },
        )
    )

    with patch("livekit.agents.beta.tools.http.send_http.http_session", return_value=mock_session):
        await tool({"id": "123", "title": "Task", "extra": "ignored"}, mock_context)

    assert mock_session.request_kwargs["json"] == {"id": "123", "title": "Task"}

    with pytest.raises(ToolError, match="Missing required parameters: id"):
        await tool({"title": "Task"}, mock_context)

    with pytest.raises(ToolError, match="Invalid type for 'title'"):
        await tool({"id": "123", "title": 10}, mock_context)


async def test_params_dataclass_support():
    """Test HTTPToolParam list builds schema and partitions args."""
    mock_context = create_mock_context()
    mock_session = MockSession()

    tool = create_http_tool(
        HTTPToolConfig(
            name="params_tool",
            url="https://api.example.com/tasks/:id",
            method="PATCH",
            params=[
                HTTPToolParam(name="id", description="task id", required=True),
                HTTPToolParam(name="title", description="new title"),
                HTTPToolParam(
                    name="status",
                    description="task status",
                    enum=["open", "closed"],
                ),
            ],
        )
    )

    with patch("livekit.agents.beta.tools.http.send_http.http_session", return_value=mock_session):
        await tool({"id": "42", "title": "Test", "status": "open", "ignored": "yes"}, mock_context)

    assert mock_session.request_kwargs["url"] == "https://api.example.com/tasks/42"
    assert mock_session.request_kwargs["json"] == {"title": "Test", "status": "open"}

    with pytest.raises(ToolError, match="Invalid value for 'status'"):
        await tool({"id": "42", "title": "Test", "status": "pending"}, mock_context)


def test_url_templating_validation():
    """Test that validation catches missing URL parameter definitions."""
    # Should raise error for missing parameter definition
    with pytest.raises(ValueError, match="URL template parameters missing from schema: id"):
        create_http_tool(
            HTTPToolConfig(
                name="missing_param_tool",
                url="https://api.example.com/tasks/:id",
                json_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "task title"}
                        # Missing "id" definition
                    },
                },
            )
        )

    # Should work when all URL params are defined
    tool = create_http_tool(
        HTTPToolConfig(
            name="valid_param_tool",
            url="https://api.example.com/tasks/:id",
            json_schema={
                "type": "object",
                "properties": {
                    "id": "the task id",
                    "title": {"type": "string", "description": "task title"},
                },
            },
        )
    )
    assert tool is not None
