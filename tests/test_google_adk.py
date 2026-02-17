"""Tests for Google ADK LLMAdapter."""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.events import Event
from google.genai import types as genai_types

from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.llm import ChatContext
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.plugins.google_adk import ADKStream, LLMAdapter


# --- Helpers ---


def _make_event(
    *,
    text: str | None = None,
    partial: bool = False,
    is_final: bool = False,
    usage: genai_types.GenerateContentResponseUsageMetadata | None = None,
    content: genai_types.Content | None = ...,  # sentinel
) -> Event:
    """Build a minimal ADK Event for testing.

    Pass ``content=None`` to create an event with no content at all.
    Omit ``content`` to auto-generate content from ``text``.
    """
    if content is ...:
        parts = [genai_types.Part(text=text)] if text else []
        content = genai_types.Content(role="model", parts=parts) if parts else None

    event = MagicMock(spec=Event)
    event.partial = partial
    event.content = content
    event.is_final_response.return_value = is_final
    event.usage_metadata = usage
    return event


async def _fake_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
    """Simulates Runner.run_async yielding a single final event."""
    yield _make_event(text="Hello from ADK", is_final=True)


async def _fake_run_async_streaming(**kwargs: Any) -> AsyncGenerator[Event, None]:
    """Simulates Runner.run_async with partial streaming then final."""
    yield _make_event(text="Hello ", partial=True)
    yield _make_event(text="world", partial=True)
    yield _make_event(text="Hello world", is_final=True)


async def _fake_run_async_with_usage(**kwargs: Any) -> AsyncGenerator[Event, None]:
    """Simulates Runner.run_async with usage metadata on final event."""
    usage = genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )
    yield _make_event(text="Hi", is_final=True, usage=usage)


async def _fake_run_async_error(**kwargs: Any) -> AsyncGenerator[Event, None]:
    """Simulates Runner.run_async that raises."""
    raise RuntimeError("ADK agent failed")
    yield  # make it a generator  # noqa: E501


async def collect_chunks(stream) -> list:
    """Collect all chunks from a stream."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


async def collect_text_chunks(stream) -> list[str]:
    """Collect only text content chunks from a stream."""
    texts = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            texts.append(chunk.delta.content)
    return texts


# --- Tests: Basic responses ---


@pytest.mark.asyncio
async def test_final_response():
    """Single final event produces one chunk with the response text."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    assert len(chunks) == 1
    assert chunks[0] == "Hello from ADK"


@pytest.mark.asyncio
async def test_streaming_partials():
    """Partial events are emitted as separate chunks."""
    adapter = _make_adapter(_fake_run_async_streaming)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    # 2 partials only (final is skipped to avoid duplicate text)
    assert len(chunks) == 2
    assert chunks[0] == "Hello "
    assert chunks[1] == "world"


@pytest.mark.asyncio
async def test_usage_metadata():
    """Token usage is reported when available on the final event."""
    adapter = _make_adapter(_fake_run_async_with_usage)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    usage_chunk = None
    async for chunk in stream:
        if chunk.usage:
            usage_chunk = chunk

    assert usage_chunk is not None
    assert usage_chunk.usage.prompt_tokens == 10
    assert usage_chunk.usage.completion_tokens == 5
    assert usage_chunk.usage.total_tokens == 15


# --- Tests: Error handling ---


@pytest.mark.asyncio
async def test_no_user_message_raises():
    """An empty chat context raises APIConnectionError."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    # No user message added
    stream = adapter.chat(chat_ctx=chat_ctx)

    with pytest.raises(APIConnectionError, match="no user message"):
        await collect_chunks(stream)


@pytest.mark.asyncio
async def test_adk_error_wraps_as_api_connection_error():
    """ADK exceptions are wrapped in APIConnectionError."""
    adapter = _make_adapter(_fake_run_async_error)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    with pytest.raises(APIConnectionError):
        await collect_chunks(stream)


@pytest.mark.asyncio
async def test_api_connection_error_passthrough():
    """APIConnectionError raised by ADK is re-raised directly, not double-wrapped."""
    from livekit.agents.types import APIConnectOptions

    async def raise_api_conn(**kwargs: Any) -> AsyncGenerator[Event, None]:
        raise APIConnectionError("direct connection failure", retryable=False)
        yield  # noqa: E501

    adapter = _make_adapter(raise_api_conn)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(
        chat_ctx=chat_ctx,
        conn_options=APIConnectOptions(max_retry=0),
    )

    with pytest.raises(APIConnectionError, match="direct connection failure"):
        await collect_chunks(stream)


@pytest.mark.asyncio
async def test_api_status_error_passthrough():
    """APIStatusError raised by ADK is re-raised directly, not double-wrapped."""
    from livekit.agents.types import APIConnectOptions

    async def raise_api_status(**kwargs: Any) -> AsyncGenerator[Event, None]:
        raise APIStatusError("status error", status_code=429, retryable=False, body=None)
        yield  # noqa: E501

    adapter = _make_adapter(raise_api_status)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(
        chat_ctx=chat_ctx,
        conn_options=APIConnectOptions(max_retry=0),
    )

    with pytest.raises(APIStatusError):
        await collect_chunks(stream)


@pytest.mark.asyncio
async def test_error_after_events_is_not_retryable():
    """An error after receiving events should be marked as not retryable."""

    async def error_after_partial(**kwargs: Any) -> AsyncGenerator[Event, None]:
        yield _make_event(text="partial", partial=True)
        raise RuntimeError("late failure")

    adapter = _make_adapter(error_after_partial)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    with pytest.raises(APIConnectionError) as exc_info:
        await collect_chunks(stream)

    # After receiving events, retryable should be False
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_error_before_events_is_retryable():
    """An error before any events should be marked as retryable."""

    async def immediate_error(**kwargs: Any) -> AsyncGenerator[Event, None]:
        raise RuntimeError("immediate failure")
        yield  # noqa: E501

    adapter = _make_adapter(immediate_error)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")

    # We need to intercept the raw _run() call to check retryable before
    # the base class retry logic gets involved. Use conn_options with no retries.
    from livekit.agents.types import APIConnectOptions

    stream = adapter.chat(
        chat_ctx=chat_ctx,
        conn_options=APIConnectOptions(max_retry=0),
    )

    with pytest.raises(APIConnectionError) as exc_info:
        await collect_chunks(stream)

    assert exc_info.value.retryable is True


# --- Tests: Message extraction ---


@pytest.mark.asyncio
async def test_extracts_latest_user_message():
    """When multiple user messages exist, the latest one is sent to ADK."""
    captured_kwargs: dict[str, Any] = {}

    async def capturing_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        captured_kwargs.update(kwargs)
        yield _make_event(text="ok", is_final=True)

    adapter = _make_adapter(capturing_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="first message")
    chat_ctx.add_message(role="assistant", content="response")
    chat_ctx.add_message(role="user", content="second message")
    stream = adapter.chat(chat_ctx=chat_ctx)
    await collect_text_chunks(stream)

    new_msg = captured_kwargs["new_message"]
    assert new_msg.parts[0].text == "second message"


@pytest.mark.asyncio
async def test_skips_non_message_items():
    """FunctionCall and FunctionCallOutput items in chat context are skipped."""
    captured_kwargs: dict[str, Any] = {}

    async def capturing_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        captured_kwargs.update(kwargs)
        yield _make_event(text="ok", is_final=True)

    adapter = _make_adapter(capturing_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="question")
    # Insert non-message items after the user message
    chat_ctx.items.append(
        FunctionCall(call_id="call_1", arguments='{"x": 1}', name="get_weather")
    )
    chat_ctx.items.append(
        FunctionCallOutput(call_id="call_1", output='{"temp": 72}', is_error=False)
    )
    stream = adapter.chat(chat_ctx=chat_ctx)
    await collect_text_chunks(stream)

    new_msg = captured_kwargs["new_message"]
    assert new_msg.parts[0].text == "question"


@pytest.mark.asyncio
async def test_only_assistant_messages_raises():
    """Chat context with only assistant messages (no user) raises."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="assistant", content="I'm helpful")
    stream = adapter.chat(chat_ctx=chat_ctx)

    with pytest.raises(APIConnectionError, match="no user message"):
        await collect_chunks(stream)


# --- Tests: Session management ---


@pytest.mark.asyncio
async def test_session_reuse():
    """The same session is reused across multiple chat() calls."""
    call_count = 0

    async def counting_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        nonlocal call_count
        call_count += 1
        yield _make_event(text=f"response {call_count}", is_final=True)

    adapter = _make_adapter(counting_run_async)

    for i in range(3):
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content=f"msg {i}")
        stream = adapter.chat(chat_ctx=chat_ctx)
        await collect_text_chunks(stream)

    assert call_count == 3
    # Only one session should have been created (cached)
    assert len(adapter._sessions) == 1


@pytest.mark.asyncio
async def test_different_users_get_different_sessions():
    """Different user_ids create separate sessions."""

    async def echo_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        yield _make_event(text="ok", is_final=True)

    adapter = _make_adapter(echo_run_async)

    for user in ["alice", "bob", "charlie"]:
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="hi")
        stream = adapter.chat(chat_ctx=chat_ctx, extra_kwargs={"user_id": user})
        await collect_text_chunks(stream)

    assert len(adapter._sessions) == 3


@pytest.mark.asyncio
async def test_session_id_in_cache_key():
    """Different session_ids for the same user create separate sessions."""

    async def echo_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        yield _make_event(text="ok", is_final=True)

    adapter = _make_adapter(echo_run_async)

    for sid in ["sess_1", "sess_2"]:
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="hi")
        stream = adapter.chat(
            chat_ctx=chat_ctx,
            extra_kwargs={"user_id": "same_user", "session_id": sid},
        )
        await collect_text_chunks(stream)

    assert len(adapter._sessions) == 2


@pytest.mark.asyncio
async def test_default_user_id():
    """Default user_id is 'livekit_user' when not provided in extra_kwargs."""
    captured_kwargs: dict[str, Any] = {}

    async def capturing_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        captured_kwargs.update(kwargs)
        yield _make_event(text="ok", is_final=True)

    adapter = _make_adapter(capturing_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    await collect_text_chunks(stream)

    assert captured_kwargs["user_id"] == "livekit_user"


@pytest.mark.asyncio
async def test_custom_user_id_and_session_id():
    """Extra kwargs user_id and session_id are passed to the runner."""
    captured_kwargs: dict[str, Any] = {}

    async def capturing_run_async(**kwargs: Any) -> AsyncGenerator[Event, None]:
        captured_kwargs.update(kwargs)
        yield _make_event(text="ok", is_final=True)

    adapter = _make_adapter(capturing_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    stream = adapter.chat(
        chat_ctx=chat_ctx,
        extra_kwargs={"user_id": "custom_user", "session_id": "custom_sess"},
    )
    await collect_text_chunks(stream)

    assert captured_kwargs["user_id"] == "custom_user"
    assert captured_kwargs["session_id"] == "test-session-id"  # from mock session


# --- Tests: Properties ---


def test_model_property_from_agent():
    """model property reads from agent.model when available."""
    adapter = _make_adapter(_fake_run_async)
    assert adapter.model == "test-model"


def test_model_property_override():
    """model_name parameter overrides agent.model."""
    agent = MagicMock()
    agent.model = "gemini-2.5-flash"

    mock_runner = MagicMock()
    mock_session_service = AsyncMock()
    adapter = LLMAdapter(
        agent=agent,
        runner=mock_runner,
        session_service=mock_session_service,
        model_name="custom-model",
    )

    assert adapter.model == "custom-model"


def test_model_fallback_no_model_attr():
    """model falls back to 'adk-agent' when agent has no model attribute."""
    agent = MagicMock(spec=[])  # spec=[] means no attributes

    mock_runner = MagicMock()
    mock_session_service = AsyncMock()
    adapter = LLMAdapter(
        agent=agent,
        runner=mock_runner,
        session_service=mock_session_service,
    )

    assert adapter.model == "adk-agent"


def test_model_fallback_none_model():
    """model falls back to 'adk-agent' when agent.model is None."""
    agent = MagicMock()
    agent.model = None

    mock_runner = MagicMock()
    mock_session_service = AsyncMock()
    adapter = LLMAdapter(
        agent=agent,
        runner=mock_runner,
        session_service=mock_session_service,
    )

    assert adapter.model == "adk-agent"


def test_provider_property():
    """provider always returns 'google-adk'."""
    adapter = _make_adapter(_fake_run_async)
    assert adapter.provider == "google-adk"


# --- Tests: Event edge cases ---


@pytest.mark.asyncio
async def test_event_with_no_content_is_skipped():
    """Events with content=None are silently skipped."""

    async def mixed_events(**kwargs: Any) -> AsyncGenerator[Event, None]:
        yield _make_event(content=None, partial=False, is_final=False)
        yield _make_event(text="answer", is_final=True)

    adapter = _make_adapter(mixed_events)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    assert chunks == ["answer"]


@pytest.mark.asyncio
async def test_event_with_empty_parts_is_skipped():
    """Events with empty parts list produce no chunks."""

    async def empty_parts(**kwargs: Any) -> AsyncGenerator[Event, None]:
        empty_content = genai_types.Content(role="model", parts=[])
        yield _make_event(content=empty_content, partial=True)
        yield _make_event(text="result", is_final=True)

    adapter = _make_adapter(empty_parts)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    assert chunks == ["result"]


@pytest.mark.asyncio
async def test_event_with_non_text_parts_skipped():
    """Parts without text (e.g., function calls) are silently skipped."""

    async def non_text_parts(**kwargs: Any) -> AsyncGenerator[Event, None]:
        # Part with no text attribute set (simulated via a real Part with only function_call)
        no_text_part = genai_types.Part(text=None)
        content = genai_types.Content(role="model", parts=[no_text_part])
        yield _make_event(content=content, is_final=True)

    adapter = _make_adapter(non_text_parts)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    assert chunks == []


@pytest.mark.asyncio
async def test_final_event_with_usage_but_no_text():
    """Final event can have usage metadata without text content."""

    async def usage_only(**kwargs: Any) -> AsyncGenerator[Event, None]:
        usage = genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=20,
            candidates_token_count=0,
            total_token_count=20,
        )
        # Final event with usage but no content
        yield _make_event(content=None, is_final=True, usage=usage)

    adapter = _make_adapter(usage_only)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    all_chunks = await collect_chunks(stream)

    usage_chunks = [c for c in all_chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.prompt_tokens == 20
    assert usage_chunks[0].usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_usage_with_cached_tokens():
    """Usage metadata correctly maps cached_content_token_count."""

    async def cached_usage(**kwargs: Any) -> AsyncGenerator[Event, None]:
        usage = genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=30,
        )
        yield _make_event(text="cached response", is_final=True, usage=usage)

    adapter = _make_adapter(cached_usage)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    usage_chunk = None
    async for chunk in stream:
        if chunk.usage:
            usage_chunk = chunk

    assert usage_chunk is not None
    assert usage_chunk.usage.prompt_cached_tokens == 30


@pytest.mark.asyncio
async def test_usage_with_none_fields():
    """Usage metadata handles None fields gracefully (defaults to 0)."""

    async def none_usage(**kwargs: Any) -> AsyncGenerator[Event, None]:
        usage = genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=None,
            candidates_token_count=None,
            total_token_count=None,
        )
        yield _make_event(text="ok", is_final=True, usage=usage)

    adapter = _make_adapter(none_usage)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    usage_chunk = None
    async for chunk in stream:
        if chunk.usage:
            usage_chunk = chunk

    assert usage_chunk is not None
    assert usage_chunk.usage.prompt_tokens == 0
    assert usage_chunk.usage.completion_tokens == 0
    assert usage_chunk.usage.total_tokens == 0


# --- Tests: Chunk structure ---


@pytest.mark.asyncio
async def test_all_chunks_have_assistant_role():
    """All emitted delta chunks have role='assistant'."""
    adapter = _make_adapter(_fake_run_async_streaming)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    async for chunk in stream:
        if chunk.delta:
            assert chunk.delta.role == "assistant"


@pytest.mark.asyncio
async def test_all_chunks_share_request_id():
    """All chunks from a single stream share the same request ID."""
    adapter = _make_adapter(_fake_run_async_streaming)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    all_chunks = await collect_chunks(stream)
    ids = {c.id for c in all_chunks}
    assert len(ids) == 1
    assert list(ids)[0].startswith("adk_")


@pytest.mark.asyncio
async def test_chat_returns_adk_stream():
    """chat() returns an ADKStream instance."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    async with adapter.chat(chat_ctx=chat_ctx) as stream:
        assert isinstance(stream, ADKStream)
        async for _ in stream:
            pass


# --- Tests: Constructor ---


def test_constructor_creates_default_session_service():
    """When no session_service is provided, InMemorySessionService is used."""
    from google.adk.sessions import InMemorySessionService

    agent = MagicMock()
    agent.name = "test_agent"

    adapter = LLMAdapter(agent=agent)

    assert isinstance(adapter._session_service, InMemorySessionService)


def test_constructor_creates_runner_from_agent():
    """When no runner is provided, one is created from the agent."""
    agent = MagicMock()
    agent.name = "test_agent"

    adapter = LLMAdapter(agent=agent)

    assert adapter._runner is not None


def test_constructor_uses_provided_runner():
    """When a runner is provided, it is used instead of creating one."""
    agent = MagicMock()
    mock_runner = MagicMock()

    adapter = LLMAdapter(agent=agent, runner=mock_runner)

    assert adapter._runner is mock_runner


def test_constructor_custom_app_name():
    """Custom app_name is stored correctly."""
    agent = MagicMock()
    mock_runner = MagicMock()

    adapter = LLMAdapter(agent=agent, runner=mock_runner, app_name="my_app")

    assert adapter._app_name == "my_app"


# --- Tests: Stream lifecycle ---


@pytest.mark.asyncio
async def test_stream_aclose():
    """Stream can be fully consumed and then closed via aclose()."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)

    # Consume fully then close
    async for _ in stream:
        pass
    await stream.aclose()


@pytest.mark.asyncio
async def test_stream_context_manager():
    """Stream works as an async context manager."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")

    async with adapter.chat(chat_ctx=chat_ctx) as stream:
        chunks = []
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)

    assert chunks == ["Hello from ADK"]


@pytest.mark.asyncio
async def test_multiple_parts_in_single_event():
    """An event with multiple text parts emits one chunk per part."""

    async def multi_part(**kwargs: Any) -> AsyncGenerator[Event, None]:
        content = genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(text="part1"),
                genai_types.Part(text="part2"),
                genai_types.Part(text="part3"),
            ],
        )
        yield _make_event(content=content, is_final=True)

    adapter = _make_adapter(multi_part)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    assert chunks == ["part1", "part2", "part3"]


@pytest.mark.asyncio
async def test_non_final_non_partial_event_skipped():
    """Events that are neither partial nor final are silently skipped."""

    async def intermediate_events(**kwargs: Any) -> AsyncGenerator[Event, None]:
        # An intermediate event (e.g., tool execution status)
        yield _make_event(text="intermediate", partial=False, is_final=False)
        yield _make_event(text="final answer", is_final=True)

    adapter = _make_adapter(intermediate_events)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_text_chunks(stream)

    # Only the final event's text should appear
    assert chunks == ["final answer"]


# --- Fixtures ---


def _make_adapter(run_async_fn) -> LLMAdapter:
    """Create an LLMAdapter with a mocked Runner."""
    agent = MagicMock()
    agent.model = "test-model"

    mock_runner = MagicMock()
    mock_runner.run_async = run_async_fn

    mock_session_service = AsyncMock()
    mock_session = MagicMock()
    mock_session.id = "test-session-id"
    mock_session_service.create_session = AsyncMock(return_value=mock_session)

    adapter = LLMAdapter(
        agent=agent,
        runner=mock_runner,
        session_service=mock_session_service,
    )
    return adapter
