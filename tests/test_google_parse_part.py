from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from livekit.plugins.google.llm import LLMStream


@pytest.fixture
def llm_stream():
    mock_llm = MagicMock()
    mock_llm._thought_signatures = {}

    with patch.object(LLMStream, "__init__", lambda self, *a, **kw: None):
        stream = LLMStream.__new__(LLMStream)
        stream._llm = mock_llm
        stream._model = "gemini-2.0-flash"
    return stream


class TestParsePartFunctionCall:
    def test_function_call_with_text_returns_none_content(self, llm_stream: LLMStream):
        part = types.Part(
            function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"}),
            text="get_weather",
        )

        chunk = llm_stream._parse_part("test-id", part)

        assert chunk is not None
        assert chunk.delta.content is None
        assert chunk.delta.tool_calls is not None
        assert len(chunk.delta.tool_calls) == 1
        tool_call = chunk.delta.tool_calls[0]
        assert tool_call.name == "get_weather"
        assert '"city": "Paris"' in tool_call.arguments

    def test_function_call_without_text_returns_none_content(self, llm_stream: LLMStream):
        part = types.Part(
            function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"}),
        )

        chunk = llm_stream._parse_part("test-id", part)

        assert chunk is not None
        assert chunk.delta.content is None
        assert len(chunk.delta.tool_calls) == 1

    def test_text_only_part_returns_text_content(self, llm_stream: LLMStream):
        part = types.Part(text="Hello world")

        chunk = llm_stream._parse_part("test-id", part)

        assert chunk is not None
        assert chunk.delta.content == "Hello world"
        assert not chunk.delta.tool_calls

    def test_empty_text_part_returns_none(self, llm_stream: LLMStream):
        part = types.Part(text="")

        chunk = llm_stream._parse_part("test-id", part)

        assert chunk is None
