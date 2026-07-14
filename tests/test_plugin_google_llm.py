from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from livekit.agents import llm
from livekit.agents.llm import ChatContext, function_tool
from livekit.agents.types import APIConnectOptions
from livekit.plugins.google.llm import LLM, LLMStream
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession
from livekit.plugins.google.tools import GoogleSearch

pytestmark = pytest.mark.plugin("google")


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


class TestCachedContentOption:
    """Verify the ``cached_content`` constructor option propagates from
    ``LLM.__init__`` through ``_LLMOptions`` and into the keyword
    arguments handed to ``GenerateContentConfig`` for every request.

    The propagation tests are ``async def`` because ``LLM.chat()`` builds
    an ``LLMStream`` whose constructor schedules a metrics-monitoring
    task on the running event loop. A sync test would raise
    ``RuntimeError: no running event loop`` before reaching the
    assertion.
    """

    @pytest.mark.asyncio
    async def test_cached_content_propagates_to_extra_kwargs(self) -> None:
        llm = LLM(model="gemini-2.5-flash", api_key="test", cached_content="cachedContents/abc123")

        stream = llm.chat(chat_ctx=ChatContext.empty())
        try:
            assert stream._extra_kwargs.get("cached_content") == "cachedContents/abc123"
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_cached_content_omitted_when_not_set(self) -> None:
        """Backward compatibility: callers that don't pass
        ``cached_content`` must produce a request config without the
        field, so existing behaviour is unchanged."""
        llm = LLM(model="gemini-2.5-flash", api_key="test")

        stream = llm.chat(chat_ctx=ChatContext.empty())
        try:
            assert "cached_content" not in stream._extra_kwargs
        finally:
            await stream.aclose()

    def test_cached_content_stored_on_opts(self) -> None:
        llm = LLM(
            model="gemini-2.5-flash",
            api_key="test",
            cached_content="projects/p/locations/us-central1/cachedContents/xyz",
        )

        assert llm._opts.cached_content == "projects/p/locations/us-central1/cachedContents/xyz"


class TestCachedContentRequestSuppression:
    """Gemini's API rejects ``generateContent`` requests that pass
    ``cached_content`` together with ``system_instruction``, ``tools``,
    or ``tool_config`` — those fields belong inside the CachedContent
    resource. The plugin therefore strips them off the outgoing request
    whenever a cache is attached. These tests run the LLMStream against
    a stubbed ``generate_content_stream`` and assert on the
    ``GenerateContentConfig`` it received.
    """

    @staticmethod
    async def _single_response_async_iter():
        """Emit one minimal-but-valid GenerateContentResponse so the
        retry layer in livekit.agents.LLM doesn't treat the stream as
        empty and re-issue the request three more times."""
        yield types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="ok")],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
        )

    @classmethod
    def _patched_stream_capture(cls) -> tuple[AsyncMock, dict]:
        captured: dict = {}

        async def fake_stream(**kwargs):
            captured["model"] = kwargs.get("model")
            captured["contents"] = kwargs.get("contents")
            captured["config"] = kwargs.get("config")
            return cls._single_response_async_iter()

        return AsyncMock(side_effect=fake_stream), captured

    @pytest.mark.asyncio
    async def test_request_omits_system_instruction_when_cached_content_set(self) -> None:
        """With a cache attached, the outgoing request must carry
        ``system_instruction=None`` — the system prompt lives in the
        cache resource and re-sending it would make Gemini return 400."""
        llm = LLM(
            model="gemini-2.5-flash",
            api_key="test",
            cached_content="cachedContents/abc123",
        )

        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="system", content="system prompt that lives in cache")
        chat_ctx.add_message(role="user", content="hi")

        fake, captured = self._patched_stream_capture()
        with patch.object(llm._client.aio.models, "generate_content_stream", fake):
            stream = llm.chat(chat_ctx=chat_ctx)
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()

        config = captured["config"]
        assert config.system_instruction is None
        assert config.cached_content == "cachedContents/abc123"

    @pytest.mark.asyncio
    async def test_request_omits_tools_when_cached_content_set(self) -> None:
        """With a cache attached, the outgoing request must NOT include
        ``tools`` even if the LLMStream was constructed with function
        tools — the tool schemas belong inside the cache resource."""

        @function_tool
        async def example_tool(query: str) -> str:
            """Look something up."""
            return query

        llm = LLM(
            model="gemini-2.5-flash",
            api_key="test",
            cached_content="cachedContents/abc123",
        )

        fake, captured = self._patched_stream_capture()
        with patch.object(llm._client.aio.models, "generate_content_stream", fake):
            stream = llm.chat(chat_ctx=ChatContext.empty(), tools=[example_tool])
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()

        config = captured["config"]
        assert config.tools is None
        assert config.tool_config is None
        assert config.cached_content == "cachedContents/abc123"

    @pytest.mark.asyncio
    async def test_request_includes_system_instruction_and_tools_when_no_cache(self) -> None:
        """Backward compatibility: without ``cached_content``, the
        request still carries ``system_instruction`` and ``tools`` as
        before. Suppression is gated strictly on the cache being set."""

        @function_tool
        async def example_tool(query: str) -> str:
            """Look something up."""
            return query

        llm = LLM(model="gemini-2.5-flash", api_key="test")

        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="system", content="system prompt sent on every request")
        chat_ctx.add_message(role="user", content="hi")

        fake, captured = self._patched_stream_capture()
        with patch.object(llm._client.aio.models, "generate_content_stream", fake):
            stream = llm.chat(chat_ctx=chat_ctx, tools=[example_tool])
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()

        config = captured["config"]
        assert config.system_instruction is not None
        assert config.tools is not None and len(config.tools) >= 1

    @pytest.mark.asyncio
    async def test_request_merges_timeout_into_caller_http_options(self) -> None:
        caller_http_options = types.HttpOptions(headers={"X-Vertex-Test": "1"})
        llm = LLM(
            model="gemini-2.5-flash",
            api_key="test",
            http_options=caller_http_options,
        )

        fake, captured = self._patched_stream_capture()
        with patch.object(llm._client.aio.models, "generate_content_stream", fake):
            stream = llm.chat(
                chat_ctx=ChatContext.empty(),
                conn_options=APIConnectOptions(timeout=7.5),
            )
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()

        config = captured["config"]
        assert config.http_options.timeout == 7500
        assert config.http_options.headers["X-Vertex-Test"] == "1"
        assert "livekit-agents/" in config.http_options.headers["x-goog-api-client"]
        assert caller_http_options.timeout is None
        assert caller_http_options.headers == {"X-Vertex-Test": "1"}


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


class TestMixedToolsRequestConstruction:
    """Combining built-in (provider) tools with function tools is only supported on the
    Gemini 3 Developer API (not Vertex). These tests drive ``chat()`` against a stubbed
    ``generate_content_stream`` and assert on the resulting request config."""

    @staticmethod
    async def _single_response_async_iter():
        yield types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="ok")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
        )

    @classmethod
    async def _capture_config(cls, llm_: LLM, **chat_kwargs):
        captured: dict = {}

        async def fake_stream(**kwargs):
            captured["config"] = kwargs.get("config")
            return cls._single_response_async_iter()

        fake = AsyncMock(side_effect=fake_stream)
        with patch.object(llm_._client.aio.models, "generate_content_stream", fake):
            stream = llm_.chat(chat_ctx=ChatContext.empty(), **chat_kwargs)
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()
        return captured["config"]

    @staticmethod
    def _weather_tool():
        @function_tool
        async def get_weather(city: str) -> str:
            """Look up the weather."""
            return city

        return get_weather

    @staticmethod
    def _has_google_search(config) -> bool:
        return bool(config.tools) and any(getattr(t, "google_search", None) for t in config.tools)

    @staticmethod
    def _has_function_declarations(config) -> bool:
        return bool(config.tools) and any(t.function_declarations for t in config.tools)

    @staticmethod
    def _server_side_enabled(config) -> bool:
        return bool(config.tool_config and config.tool_config.include_server_side_tool_invocations)

    @pytest.mark.asyncio
    async def test_dev_api_enables_server_side_invocations(self) -> None:
        """Gemini 3 Developer API: both tool types are sent and the circulation flag is set."""
        llm_ = LLM(model="gemini-3-flash-preview", api_key="test")
        config = await self._capture_config(llm_, tools=[self._weather_tool(), GoogleSearch()])

        assert self._server_side_enabled(config)
        assert self._has_function_declarations(config)
        assert self._has_google_search(config)

    @pytest.mark.asyncio
    async def test_auto_mode_kept_for_mixed_tools(self) -> None:
        """``tool_choice='auto'`` is passed through as AUTO alongside the circulation flag.
        Verified live against the Gemini 3 Developer API: AUTO + built-in tools is accepted,
        so no VALIDATED upgrade is applied."""
        llm_ = LLM(model="gemini-3-flash-preview", api_key="test")
        config = await self._capture_config(
            llm_, tools=[self._weather_tool(), GoogleSearch()], tool_choice="auto"
        )

        assert self._server_side_enabled(config)
        assert (
            config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO
        )

    @pytest.mark.asyncio
    async def test_non_gemini_3_drops_provider_tool(self) -> None:
        """Below Gemini 3 the provider tool is dropped and no flag is set."""
        llm_ = LLM(model="gemini-2.5-flash", api_key="test")
        config = await self._capture_config(llm_, tools=[self._weather_tool(), GoogleSearch()])

        assert not self._server_side_enabled(config)
        assert self._has_function_declarations(config)
        assert not self._has_google_search(config)

    @pytest.mark.asyncio
    async def test_vertex_gemini_3_drops_provider_tool(self) -> None:
        """Vertex AI does not support mixing, even on Gemini 3: provider tool is dropped."""
        from google.auth.credentials import AnonymousCredentials

        llm_ = LLM(
            model="gemini-3-flash-preview",
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=AnonymousCredentials(),
        )
        assert llm_._client.vertexai is True

        config = await self._capture_config(llm_, tools=[self._weather_tool(), GoogleSearch()])

        assert not self._server_side_enabled(config)
        assert self._has_function_declarations(config)
        assert not self._has_google_search(config)

    @pytest.mark.asyncio
    async def test_provider_only_does_not_set_flag(self) -> None:
        """The circulation flag requires BOTH function and provider tools; a provider tool
        alone is sent without it."""
        llm_ = LLM(model="gemini-3-flash-preview", api_key="test")
        config = await self._capture_config(llm_, tools=[GoogleSearch()])

        assert not self._server_side_enabled(config)
        assert self._has_google_search(config)

    @pytest.mark.asyncio
    async def test_cached_content_via_extra_kwargs_suppresses_tools(self) -> None:
        """cached_content passed through ``extra_kwargs`` (not just the constructor) must
        still skip building tools/tool_config, so the request isn't rejected for combining
        them with a cache."""
        llm_ = LLM(model="gemini-3-flash-preview", api_key="test")
        config = await self._capture_config(
            llm_,
            tools=[self._weather_tool(), GoogleSearch()],
            extra_kwargs={"cached_content": "cachedContents/abc123"},
        )

        assert config.cached_content == "cachedContents/abc123"
        assert config.tools is None
        assert config.tool_config is None
