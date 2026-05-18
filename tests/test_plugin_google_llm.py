from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from livekit.agents import llm
from livekit.plugins.google.llm import LLM, LLMStream
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession


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


class TestMediaResolution:
    def test_llm_media_resolution_is_passed_to_stream_kwargs(self):
        model = LLM(
            api_key="test-api-key",
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        )

        with patch.object(
            LLMStream,
            "__init__",
            lambda self, *a, **kw: self.__dict__.update(_extra_kwargs=kw["extra_kwargs"]),
        ):
            stream = model.chat(chat_ctx=llm.ChatContext.empty())

        assert (
            stream._extra_kwargs["media_resolution"] == types.MediaResolution.MEDIA_RESOLUTION_LOW
        )

    def test_llm_media_resolution_is_omitted_by_default(self):
        model = LLM(api_key="test-api-key")

        with patch.object(
            LLMStream,
            "__init__",
            lambda self, *a, **kw: self.__dict__.update(_extra_kwargs=kw["extra_kwargs"]),
        ):
            stream = model.chat(chat_ctx=llm.ChatContext.empty())

        assert "media_resolution" not in stream._extra_kwargs

    def test_realtime_media_resolution_is_passed_to_connect_config(self):
        model = RealtimeModel(
            api_key="test-api-key",
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        )
        session = RealtimeSession.__new__(RealtimeSession)
        session._opts = model._opts
        session._tools = llm.ToolContext.empty()
        session._realtime_model = model
        session._session_resumption_handle = None

        config = session._build_connect_config()

        assert config.generation_config
        assert (
            config.generation_config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW
        )

    def test_realtime_media_resolution_is_unset_by_default(self):
        model = RealtimeModel(api_key="test-api-key")
        session = RealtimeSession.__new__(RealtimeSession)
        session._opts = model._opts
        session._tools = llm.ToolContext.empty()
        session._realtime_model = model
        session._session_resumption_handle = None

        config = session._build_connect_config()

        assert config.generation_config
        assert config.generation_config.media_resolution is None
