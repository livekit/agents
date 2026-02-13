"""Tests for Google ADK LLMAdapter."""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.events import Event
from google.genai import types as genai_types

from livekit.agents import APIConnectionError
from livekit.agents.llm import ChatContext
from livekit.plugins.google_adk import LLMAdapter


# --- Helpers ---


def _make_event(
    *,
    text: str | None = None,
    partial: bool = False,
    is_final: bool = False,
    usage: genai_types.GenerateContentResponseUsageMetadata | None = None,
) -> Event:
    """Build a minimal ADK Event for testing."""
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


async def collect_chunks(stream) -> list[str]:
    """Collect all text content chunks from a stream."""
    chunks = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    return chunks


# --- Tests ---


@pytest.mark.asyncio
async def test_final_response():
    """Single final event produces one chunk with the response text."""
    adapter = _make_adapter(_fake_run_async)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert len(chunks) == 1
    assert chunks[0] == "Hello from ADK"


@pytest.mark.asyncio
async def test_streaming_partials():
    """Partial events are emitted as separate chunks."""
    adapter = _make_adapter(_fake_run_async_streaming)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    # 2 partials + 1 final
    assert len(chunks) == 3
    assert chunks[0] == "Hello "
    assert chunks[1] == "world"
    assert chunks[2] == "Hello world"


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

    # The base LLMStream retries retryable errors, so the final exception
    # may be the aggregated retry-exhaustion message rather than the original.
    with pytest.raises(APIConnectionError):
        await collect_chunks(stream)


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
    await collect_chunks(stream)

    # The new_message sent to ADK should contain the latest user text
    new_msg = captured_kwargs["new_message"]
    assert new_msg.parts[0].text == "second message"


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
        await collect_chunks(stream)

    assert call_count == 3
    # Only one session should have been created (cached)
    assert len(adapter._sessions) == 1


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


def test_provider_property():
    """provider always returns 'google-adk'."""
    adapter = _make_adapter(_fake_run_async)
    assert adapter.provider == "google-adk"


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
