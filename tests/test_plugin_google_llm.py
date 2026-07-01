from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from livekit.agents import llm, utils
from livekit.agents.llm import ChatContext, function_tool
from livekit.agents.types import APIConnectOptions
from livekit.plugins.google.llm import LLM, LLMStream
from livekit.plugins.google.realtime import realtime_api as google_realtime
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession

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


class TestRealtimeToolCallAudioDrain:
    @staticmethod
    def _audio_content(data: bytes) -> types.LiveServerContent:
        return types.LiveServerContent(
            model_turn=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=data,
                            mime_type="audio/pcm;rate=24000",
                        )
                    )
                ]
            )
        )

    @staticmethod
    def _tool_call(call_id: str = "call-1", name: str = "end_call") -> types.LiveServerToolCall:
        return types.LiveServerToolCall(
            function_calls=[types.FunctionCall(id=call_id, name=name, args={})]
        )

    @staticmethod
    def _new_session_with_generation() -> tuple[
        RealtimeSession, google_realtime._ResponseGeneration
    ]:
        session = TestRealtimeToolCallAudioDrain._new_session()

        gen = google_realtime._ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            input_id="GI_test",
            response_id="GR_test",
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan(),
        )
        session._current_generation = gen
        return session, gen

    @staticmethod
    def _new_session(*, modalities: list[types.Modality] | None = None) -> RealtimeSession:
        model = RealtimeModel(
            api_key="test-api-key",
            modalities=modalities if modalities is not None else [types.Modality.AUDIO],
        )
        session = RealtimeSession.__new__(RealtimeSession)
        llm.RealtimeSession.__init__(session, model)
        session._opts = model._opts
        session._chat_ctx = llm.ChatContext.empty()
        session._msg_ch = utils.aio.Chan()
        session._session_should_close = asyncio.Event()
        session._session_lock = asyncio.Lock()
        session._main_atask = None
        session._active_session = None
        session._response_created_futures = {}
        session._pending_generation_fut = None
        session._rejected_tool_calls = 0
        session._current_generation = None
        return session

    @staticmethod
    async def _wait_for_generation_done(gen: google_realtime._ResponseGeneration) -> None:
        if gen._audio_drain_atask and not gen._audio_drain_atask.done():
            await gen._audio_drain_atask

    @staticmethod
    def _drain_function_calls(
        gen: google_realtime._ResponseGeneration,
    ) -> list[llm.FunctionCall]:
        calls = []
        while not gen.function_ch.empty():
            calls.append(gen.function_ch.recv_nowait())
        return calls

    @pytest.mark.asyncio
    async def test_tool_call_keeps_audio_open_until_trailing_audio_drains(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.01)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session, gen = self._new_session_with_generation()

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call())

        assert not gen.function_ch.closed
        assert not gen.audio_ch.closed
        call = gen.function_ch.recv_nowait()
        assert call.call_id == "call-1"

        session._handle_server_content(self._audio_content(b"\x02" * 960))
        assert gen.audio_ch.qsize() == 2

        await asyncio.wait_for(self._wait_for_generation_done(gen), timeout=1.0)

        assert gen.audio_ch.closed
        assert gen.message_ch.closed
        assert gen.function_ch.closed
        assert gen._done

    @pytest.mark.asyncio
    async def test_repeated_tool_call_uses_same_generation_while_audio_drains(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.01)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session, first_gen = self._new_session_with_generation()

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call(call_id="call-1", name="first_tool"))

        assert not first_gen.function_ch.closed
        assert not first_gen.audio_ch.closed

        session._handle_tool_calls(self._tool_call(call_id="call-2", name="second_tool"))

        assert session._current_generation is first_gen
        assert not first_gen.audio_ch.closed

        calls = self._drain_function_calls(first_gen)
        assert [call.call_id for call in calls] == ["call-1", "call-2"]
        assert [call.name for call in calls] == ["first_tool", "second_tool"]

        await asyncio.wait_for(self._wait_for_generation_done(first_gen), timeout=1.0)

        assert first_gen.audio_ch.closed
        assert first_gen.message_ch.closed
        assert first_gen.function_ch.closed
        assert first_gen._done

    @pytest.mark.asyncio
    async def test_trailing_audio_after_repeated_tool_call_stays_on_draining_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.01)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session, gen = self._new_session_with_generation()

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call(call_id="call-1", name="first_tool"))
        session._handle_tool_calls(self._tool_call(call_id="call-2", name="second_tool"))
        session._handle_server_content(self._audio_content(b"\x02" * 960))

        assert session._current_generation is gen
        assert gen.audio_ch.qsize() == 2
        assert not gen.audio_ch.closed

        await asyncio.wait_for(self._wait_for_generation_done(gen), timeout=1.0)

        assert gen.audio_ch.closed
        assert gen.message_ch.closed
        assert gen.function_ch.closed

    @pytest.mark.asyncio
    async def test_repeated_tool_call_refreshes_audio_drain_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.02)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session, gen = self._new_session_with_generation()

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call(call_id="call-1", name="first_tool"))
        await asyncio.sleep(0.015)

        session._handle_tool_calls(self._tool_call(call_id="call-2", name="second_tool"))
        await asyncio.sleep(0.01)

        assert not gen.audio_ch.closed
        assert not gen.function_ch.closed

        session._handle_server_content(self._audio_content(b"\x02" * 960))
        assert gen.audio_ch.qsize() == 2

        await asyncio.wait_for(self._wait_for_generation_done(gen), timeout=1.0)

        assert gen.audio_ch.closed
        assert gen.function_ch.closed

    @pytest.mark.asyncio
    async def test_repeated_tool_calls_are_emitted_on_generation_event_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.01)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session = self._new_session()
        generation_events: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generation_events.append)

        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call(call_id="call-1", name="first_tool"))
        session._handle_tool_calls(self._tool_call(call_id="call-2", name="second_tool"))
        await asyncio.wait_for(self._wait_for_generation_done(gen), timeout=1.0)

        assert len(generation_events) == 1
        calls = [call async for call in generation_events[0].function_stream]
        assert [call.call_id for call in calls] == ["call-1", "call-2"]
        assert [call.name for call in calls] == ["first_tool", "second_tool"]

    def test_tool_call_finalizes_immediately_when_audio_is_already_closed(self) -> None:
        session, gen = self._new_session_with_generation()
        gen.audio_ch.close()

        session._handle_tool_calls(self._tool_call())

        assert gen._audio_drain_atask is None
        assert gen.function_ch.closed
        assert gen.message_ch.closed
        assert gen._done

    def test_tool_call_finalizes_immediately_for_text_only_modality(self) -> None:
        session = self._new_session(modalities=[types.Modality.TEXT])
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_tool_calls(self._tool_call())

        assert gen._audio_drain_atask is None
        assert gen.function_ch.closed
        assert gen.message_ch.closed
        assert gen._done

    @pytest.mark.asyncio
    async def test_audio_modality_waits_for_first_trailing_audio_after_tool_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.01)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session, gen = self._new_session_with_generation()

        session._handle_tool_calls(self._tool_call())

        assert not gen.audio_ch.closed
        assert not gen.function_ch.closed

        session._handle_server_content(self._audio_content(b"\x01" * 960))

        assert gen.audio_ch.qsize() == 1

        await asyncio.wait_for(self._wait_for_generation_done(gen), timeout=1.0)

        assert gen.audio_ch.closed
        assert gen.function_ch.closed
        assert gen._done

    @pytest.mark.asyncio
    async def test_chained_tool_call_does_not_emit_unpaired_speech_stopped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.01)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 0.2)
        session = self._new_session()
        speech_events: list[str] = []
        session.on("input_speech_started", lambda _: speech_events.append("started"))
        session.on("input_speech_stopped", lambda _: speech_events.append("stopped"))

        session._start_new_generation()
        first_gen = session._current_generation
        assert first_gen is not None

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call(call_id="call-1", name="first_tool"))
        session._handle_tool_calls(self._tool_call(call_id="call-2", name="second_tool"))

        await self._wait_for_generation_done(first_gen)

        assert first_gen._done
        assert speech_events == ["started", "stopped"]

    @pytest.mark.asyncio
    async def test_close_finalizes_draining_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_QUIESCENCE", 0.2)
        monkeypatch.setattr(google_realtime, "TOOL_CALL_AUDIO_DRAIN_TIMEOUT", 1.0)
        session = self._new_session()

        session._start_new_generation()
        first_gen = session._current_generation
        assert first_gen is not None

        session._handle_server_content(self._audio_content(b"\x01" * 960))
        session._handle_tool_calls(self._tool_call(call_id="call-1", name="first_tool"))
        session._handle_tool_calls(self._tool_call(call_id="call-2", name="second_tool"))

        await session.aclose()
        await asyncio.sleep(0)

        assert first_gen._done
        assert first_gen.audio_ch.closed
        assert first_gen.message_ch.closed
