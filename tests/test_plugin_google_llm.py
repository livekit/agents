from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from livekit.agents import llm
from livekit.agents.llm import ChatContext, function_tool
from livekit.agents.types import APIConnectOptions
from livekit.plugins.google.llm import LLM, LLMStream
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession
from livekit.plugins.google.tools import GoogleSearch

pytestmark = pytest.mark.plugin("google")


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return city


def _search_parts() -> list[types.Part]:
    return [
        types.Part(
            thought_signature=b"search-signature",
            tool_call=types.ToolCall(
                id="search-1",
                tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                args={"queries": ["northernmost city in the United States"]},
            ),
        ),
        types.Part(
            thought_signature=b"search-signature",
            tool_response=types.ToolResponse(
                id="search-1",
                tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                response={"search_suggestions": "Utqiagvik"},
            ),
        ),
    ]


def _mixed_google_content() -> types.Content:
    return types.Content(
        role="model",
        parts=[
            *_search_parts(),
            types.Part(
                thought_signature=b"function-signature",
                function_call=types.FunctionCall(
                    id="weather-1", name="get_weather", args={"city": "Utqiagvik, Alaska"}
                ),
            ),
            types.Part(text="Reasoning that should not be surfaced as output.", thought=True),
        ],
    )


def _content_dict(content: types.Content) -> dict:
    return content.model_dump(mode="json", exclude_none=True)


def _google_content_extra(content: types.Content) -> dict:
    return {"google": {"content": _content_dict(content)}}


def _response(content: types.Content) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=content,
                finish_reason=types.FinishReason.STOP,
            )
        ],
    )


async def _capture_request(model: LLM, **chat_kwargs) -> types.GenerateContentConfig:
    captured: dict = {}

    async def stream(**kwargs):
        captured["config"] = kwargs["config"]
        yield _response(types.Content(role="model", parts=[types.Part(text="ok")]))

    chat_kwargs.setdefault("chat_ctx", ChatContext.empty())
    with patch.object(
        model._client.aio.models, "generate_content_stream", AsyncMock(side_effect=stream)
    ):
        await model.chat(**chat_kwargs).collect()

    return captured["config"]


async def _collect_response(model: LLM, content: types.Content, **chat_kwargs):
    chat_kwargs.setdefault("chat_ctx", ChatContext.empty())

    async def stream(**kwargs):
        yield _response(content)

    with patch.object(
        model._client.aio.models, "generate_content_stream", AsyncMock(side_effect=stream)
    ):
        return await model.chat(**chat_kwargs).collect()


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_function_call_surfaced_as_tool_call(self) -> None:
        content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"})
                )
            ],
        )
        model = LLM(model="gemini-2.0-flash", api_key="test")
        response = await _collect_response(model, content, tools=[get_weather])

        assert response.text == ""
        [tool_call] = response.tool_calls
        assert tool_call.name == "get_weather"
        assert '"city": "Paris"' in tool_call.arguments
        assert not tool_call.extra

    @pytest.mark.asyncio
    async def test_text_surfaced_as_content(self) -> None:
        content = types.Content(role="model", parts=[types.Part(text="Hello world")])
        model = LLM(model="gemini-2.0-flash", api_key="test")
        response = await _collect_response(model, content)

        assert response.text == "Hello world"
        assert not response.tool_calls

    @pytest.mark.asyncio
    async def test_empty_text_part_counts_as_a_response(self) -> None:
        content = types.Content(role="model", parts=[types.Part(text="")])
        model = LLM(model="gemini-2.0-flash", api_key="test")
        response = await _collect_response(model, content)

        assert response.text == ""
        assert not response.tool_calls


class TestServerSideToolInvocations:
    @pytest.mark.asyncio
    async def test_mixed_response_preserves_content_on_function_call(self) -> None:
        content = _mixed_google_content()
        model = LLM(model="gemini-3.5-flash", api_key="test")
        response = await _collect_response(model, content, tools=[get_weather, GoogleSearch()])

        assert response.text == ""
        [tool_call] = response.tool_calls
        assert tool_call.call_id == "weather-1"
        assert tool_call.extra == _google_content_extra(content)

    def test_formatter_replays_function_call_content_then_response(self) -> None:
        content = _mixed_google_content()
        chat_ctx = ChatContext.empty()
        chat_ctx.items.append(
            llm.FunctionCall(
                call_id="weather-1",
                name="get_weather",
                arguments="{}",
                extra=_google_content_extra(content),
            )
        )
        chat_ctx.items.append(
            llm.FunctionCallOutput(
                call_id="weather-1", name="get_weather", output="22F", is_error=False
            )
        )

        turns, _ = chat_ctx.to_provider_format("google")

        assert turns == [
            _content_dict(content),
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "weather-1",
                            "name": "get_weather",
                            "response": {"output": "22F"},
                        }
                    }
                ],
            },
        ]

    def test_formatter_replays_message_content_as_model_turn(self) -> None:
        content = types.Content(
            role="model",
            parts=[
                *_search_parts(),
                types.Part(text="The northernmost city is Utqiagvik, Alaska."),
            ],
        )
        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(
            role="assistant",
            content="The northernmost city is Utqiagvik, Alaska.",
            extra=_google_content_extra(content),
        )
        chat_ctx.add_message(role="user", content="thanks")

        turns, _ = chat_ctx.to_provider_format("google")

        assert turns == [
            _content_dict(content),
            {"role": "user", "parts": [{"text": "thanks"}]},
        ]

    @pytest.mark.asyncio
    async def test_mixed_tools_run_server_side(self) -> None:
        model = LLM(model="gemini-3.5-flash", api_key="test")

        config = await _capture_request(model, tools=[get_weather, GoogleSearch()])

        assert config.tool_config.include_server_side_tool_invocations is True
        assert config.tool_config.function_calling_config is None
        assert any(tool.function_declarations for tool in config.tools)
        assert any(tool.google_search for tool in config.tools)

    @pytest.mark.asyncio
    async def test_required_tool_choice_preserved_with_provider_tools(self) -> None:
        model = LLM(model="gemini-3.5-flash", api_key="test")

        config = await _capture_request(
            model, tools=[get_weather, GoogleSearch()], tool_choice="required"
        )

        assert config.tool_config.include_server_side_tool_invocations is True
        assert (
            config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.ANY
        )
        assert config.tool_config.function_calling_config.allowed_function_names == ["get_weather"]

    @pytest.mark.asyncio
    async def test_specific_function_tool_choice_preserved_with_provider_tools(self) -> None:
        model = LLM(model="gemini-3.5-flash", api_key="test")

        config = await _capture_request(
            model,
            tools=[get_weather, GoogleSearch()],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        assert config.tool_config.include_server_side_tool_invocations is True
        assert (
            config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.ANY
        )
        assert config.tool_config.function_calling_config.allowed_function_names == ["get_weather"]

    @pytest.mark.asyncio
    async def test_provider_tools_alone_still_circulate(self) -> None:
        model = LLM(model="gemini-3.5-flash", api_key="test")

        config = await _capture_request(model, tools=[GoogleSearch()])

        assert config.tool_config.include_server_side_tool_invocations is True
        assert config.tool_config.function_calling_config is None
        assert any(tool.google_search for tool in config.tools)
        assert not any(tool.function_declarations for tool in config.tools)

    @pytest.mark.asyncio
    async def test_provider_tools_alone_below_gemini_3_skip_circulation(self) -> None:
        model = LLM(model="gemini-2.5-flash", api_key="test")

        config = await _capture_request(model, tools=[GoogleSearch()])

        assert config.tool_config is None
        assert any(tool.google_search for tool in config.tools)

    @pytest.mark.asyncio
    async def test_mixed_tools_keep_retrieval_config(self) -> None:
        model = LLM(
            model="gemini-3.5-flash",
            api_key="test",
            retrieval_config=types.RetrievalConfig(language_code="en"),
        )

        config = await _capture_request(model, tools=[get_weather, GoogleSearch()])

        assert config.tool_config.include_server_side_tool_invocations is True
        assert config.tool_config.function_calling_config is None
        assert config.tool_config.retrieval_config.language_code == "en"

    @pytest.mark.asyncio
    async def test_tool_choice_none_drops_provider(self) -> None:
        model = LLM(model="gemini-3.5-flash", api_key="test")

        config = await _capture_request(
            model, tools=[get_weather, GoogleSearch()], tool_choice="none"
        )

        assert (
            config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE
        )
        assert any(tool.function_declarations for tool in config.tools)
        assert not any(tool.google_search for tool in config.tools)

    @pytest.mark.asyncio
    async def test_tool_choice_none_sends_no_provider_tools(self) -> None:
        model = LLM(model="gemini-3.5-flash", api_key="test")

        config = await _capture_request(model, tools=[GoogleSearch()], tool_choice="none")

        assert config.tools is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm_kwargs",
        [
            {"model": "gemini-2.5-flash", "api_key": "test"},
            {"model": "gemini-3.5-flash", "vertexai": True, "project": "p", "location": "us-east1"},
        ],
        ids=["older-model", "vertex"],
    )
    async def test_drops_provider_tools_when_combination_unsupported(
        self, llm_kwargs: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = LLM(**llm_kwargs)

        config = await _capture_request(model, tools=[get_weather, GoogleSearch()])

        assert config.tool_config is None
        assert any(tool.function_declarations for tool in config.tools)
        assert not any(tool.google_search for tool in config.tools)
        assert "dropping provider tools" in caplog.text

    def test_formatter_replays_parallel_function_calls(self) -> None:
        content = types.Content(
            role="model",
            parts=[
                *_search_parts(),
                types.Part(
                    thought_signature=b"weather-signature",
                    function_call=types.FunctionCall(
                        id="weather-1", name="get_weather", args={"city": "Utqiagvik, Alaska"}
                    ),
                ),
                types.Part(
                    thought_signature=b"time-signature",
                    function_call=types.FunctionCall(
                        id="time-1", name="get_time", args={"city": "Utqiagvik, Alaska"}
                    ),
                ),
            ],
        )
        chat_ctx = ChatContext.empty()
        chat_ctx.items.append(
            llm.FunctionCall(
                call_id="weather-1",
                name="get_weather",
                arguments="{}",
                group_id="g1",
                extra=_google_content_extra(content),
            )
        )
        chat_ctx.items.append(
            llm.FunctionCall(
                call_id="time-1",
                name="get_time",
                arguments="{}",
                group_id="g1",
            )
        )
        chat_ctx.items.append(
            llm.FunctionCallOutput(
                call_id="weather-1", name="get_weather", output="22F", is_error=False
            )
        )
        chat_ctx.items.append(
            llm.FunctionCallOutput(call_id="time-1", name="get_time", output="3am", is_error=False)
        )

        turns, _ = chat_ctx.to_provider_format("google")

        assert turns == [
            _content_dict(content),
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "weather-1",
                            "name": "get_weather",
                            "response": {"output": "22F"},
                        }
                    },
                    {
                        "function_response": {
                            "id": "time-1",
                            "name": "get_time",
                            "response": {"output": "3am"},
                        }
                    },
                ],
            },
        ]


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
    async def test_cached_content_drops_mixed_tools(self, caplog: pytest.LogCaptureFixture) -> None:
        model = LLM(
            model="gemini-3.5-flash",
            api_key="test",
            cached_content="cachedContents/abc123",
        )

        config = await _capture_request(model, tools=[get_weather, GoogleSearch()])

        assert config.tools is None
        assert config.tool_config is None
        assert config.cached_content == "cachedContents/abc123"
        assert "CachedContent resource" in caplog.text

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
