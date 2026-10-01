"""Tests for the livekit-plugins-dialogflow plugin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import llm
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from livekit.plugins import dialogflow


@pytest.fixture(autouse=True)
def _mock_sessions_client():
    """Mock SessionsAsyncClient to prevent real gRPC connections in all tests."""
    with patch("livekit.plugins.dialogflow.llm.SessionsAsyncClient"):
        yield


# --- LLM class tests ---


def test_llm_instantiation():
    """LLM can be instantiated with required args."""
    llm_instance = dialogflow.LLM(
        project_id="test-project",
        agent_id="test-agent-id",
        location="us-central1",
    )
    assert llm_instance.model == "dialogflow-cx/test-agent-id"
    assert llm_instance.provider == "Google Dialogflow CX"


def test_llm_env_var_fallback(monkeypatch):
    """project_id and location fall back to environment variables."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "eu-west1")

    llm_instance = dialogflow.LLM(agent_id="agent-1")
    assert llm_instance._opts.project_id == "env-project"
    assert llm_instance._opts.location == "eu-west1"


def test_llm_env_var_location_default(monkeypatch):
    """Location defaults to 'global' when not provided."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    llm_instance = dialogflow.LLM(agent_id="agent-1")
    assert llm_instance._opts.location == "global"


def test_llm_missing_project_id(monkeypatch):
    """Raises ValueError when project_id is not set anywhere."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="project_id must be provided"):
        dialogflow.LLM(agent_id="agent-1")


def test_llm_explicit_args_override_env(monkeypatch):
    """Explicit constructor args take precedence over env vars."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")

    llm_instance = dialogflow.LLM(
        project_id="explicit-project",
        location="explicit-location",
        agent_id="agent-1",
    )
    assert llm_instance._opts.project_id == "explicit-project"
    assert llm_instance._opts.location == "explicit-location"


def test_llm_default_options():
    """Default values for optional parameters are correct."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    assert llm_instance._opts.language_code == "en"
    assert llm_instance._opts.environment_id is None
    assert llm_instance._opts.session_ttl == 3600


def test_llm_is_subclass():
    """LLM is a proper subclass of llm.LLM."""
    assert issubclass(dialogflow.LLM, llm.LLM)


def test_llm_stream_is_subclass():
    """DialogflowLLMStream is a proper subclass of llm.LLMStream."""
    assert issubclass(dialogflow.LLMStream, llm.LLMStream)


# --- chat() method tests ---


@pytest.mark.asyncio
async def test_chat_returns_llm_stream():
    """chat() returns a DialogflowLLMStream instance."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    assert isinstance(stream, llm.LLMStream)
    assert isinstance(stream, dialogflow.LLMStream)
    await stream.aclose()


@pytest.mark.asyncio
async def test_chat_passes_extra_kwargs():
    """extra_kwargs are passed through to the stream."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx, extra_kwargs={"session_id": "my-session"})
    assert stream._extra_kwargs["session_id"] == "my-session"
    await stream.aclose()


# --- _run() method tests with mocked Dialogflow API ---


def _make_mock_response(texts: list[str], payload=None, match_type=None):
    """Create a mock DetectIntentResponse."""
    response = MagicMock()

    messages = []
    for text in texts:
        msg = MagicMock()
        msg.text.text = [text]
        msg.payload = None
        messages.append(msg)

    if payload is not None:
        payload_msg = MagicMock()
        payload_msg.text = None
        payload_msg.payload = payload
        messages.append(payload_msg)

    response.query_result.response_messages = messages
    response.query_result.intent_detection_confidence = 0.95
    response.query_result.match.match_type.name = match_type or "INTENT"
    response.query_result.match.intent.display_name = "test.intent"
    response.query_result.current_page.display_name = "Start Page"
    return response


@pytest.mark.asyncio
async def test_run_emits_response_and_usage():
    """_run() emits a text ChatChunk and a usage ChatChunk."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(
        return_value=_make_mock_response(["Hello!", "How can I help?"])
    )

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hi there")

    stream = llm_instance.chat(chat_ctx=chat_ctx)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # Should have at least 2 chunks: one with delta, one with usage
    assert len(chunks) >= 2

    # First chunk has the response text
    text_chunk = chunks[0]
    assert text_chunk.delta is not None
    assert text_chunk.delta.role == "assistant"
    assert text_chunk.delta.content == "Hello! How can I help?"

    # Check metadata in extra
    assert text_chunk.delta.extra is not None
    assert text_chunk.delta.extra["intent_detection_confidence"] == 0.95
    assert text_chunk.delta.extra["match_type"] == "INTENT"
    assert text_chunk.delta.extra["matched_intent"] == "test.intent"
    assert text_chunk.delta.extra["current_page"] == "Start Page"

    # Last chunk has usage
    usage_chunk = chunks[1]
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.completion_tokens == 0
    assert usage_chunk.usage.prompt_tokens == 0
    assert usage_chunk.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_run_custom_payload():
    """_run() includes custom payloads in the extra dict."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    payload = {"richContent": [{"type": "button", "text": "Click me"}]}
    llm_instance._client.detect_intent = AsyncMock(
        return_value=_make_mock_response(["Here's a button:"], payload=payload)
    )

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Show me a button")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    text_chunk = chunks[0]
    assert text_chunk.delta.extra is not None
    assert "custom_payloads" in text_chunk.delta.extra
    assert len(text_chunk.delta.extra["custom_payloads"]) == 1


@pytest.mark.asyncio
async def test_run_no_user_message():
    """_run() raises APIStatusError when no user message exists."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()

    # Chat context with only a system message, no user message
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="You are a bot")

    stream = llm_instance.chat(chat_ctx=chat_ctx)

    # The stream should raise; collect will propagate the error
    with pytest.raises(APIStatusError, match="no user message"):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_run_session_id_from_extra_kwargs():
    """Session ID can be provided via extra_kwargs."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent", location="global")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(return_value=_make_mock_response(["OK"]))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx, extra_kwargs={"session_id": "custom-session-123"})
    async for _ in stream:
        pass

    # Verify detect_intent was called with the custom session ID in the path
    call_args = llm_instance._client.detect_intent.call_args
    request = call_args.kwargs["request"]
    assert "custom-session-123" in request.session


@pytest.mark.asyncio
async def test_run_environment_id_in_session_path():
    """Session path includes environment_id when set."""
    llm_instance = dialogflow.LLM(
        project_id="proj",
        agent_id="agent",
        location="us-central1",
        environment_id="production",
    )
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(return_value=_make_mock_response(["OK"]))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx, extra_kwargs={"session_id": "sess-1"})
    async for _ in stream:
        pass

    call_args = llm_instance._client.detect_intent.call_args
    request = call_args.kwargs["request"]
    expected = (
        "projects/proj/locations/us-central1/agents/agent/environments/production/sessions/sess-1"
    )
    assert request.session == expected


@pytest.mark.asyncio
async def test_run_no_environment_id_in_session_path():
    """Session path does NOT include environments when environment_id is None."""
    llm_instance = dialogflow.LLM(
        project_id="proj",
        agent_id="agent",
        location="global",
    )
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(return_value=_make_mock_response(["OK"]))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx, extra_kwargs={"session_id": "sess-2"})
    async for _ in stream:
        pass

    call_args = llm_instance._client.detect_intent.call_args
    request = call_args.kwargs["request"]
    expected = "projects/proj/locations/global/agents/agent/sessions/sess-2"
    assert request.session == expected
    assert "environments" not in request.session


# --- Error handling tests ---


@pytest.mark.asyncio
async def test_error_resource_exhausted():
    """ResourceExhausted maps to APIStatusError 429."""
    from google.api_core.exceptions import ResourceExhausted

    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(side_effect=ResourceExhausted("quota exceeded"))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_error_service_unavailable():
    """ServiceUnavailable maps to APIStatusError 503."""
    from google.api_core.exceptions import ServiceUnavailable

    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(side_effect=ServiceUnavailable("down"))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_error_invalid_argument():
    """InvalidArgument maps to APIStatusError 400 (non-retryable)."""
    from google.api_core.exceptions import InvalidArgument

    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(side_effect=InvalidArgument("bad request"))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_error_unexpected_exception():
    """Unexpected exceptions map to APIConnectionError."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(side_effect=RuntimeError("something broke"))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_run_extracts_latest_user_message():
    """_run() uses the LAST user message, not the first."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(return_value=_make_mock_response(["Got it"]))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="First message")
    chat_ctx.add_message(role="assistant", content="Response")
    chat_ctx.add_message(role="user", content="Second message")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    async for _ in stream:
        pass

    # Verify the last user message was sent
    call_args = llm_instance._client.detect_intent.call_args
    request = call_args.kwargs["request"]
    assert request.query_input.text.text == "Second message"


@pytest.mark.asyncio
async def test_run_language_code():
    """Language code from LLM options is passed to QueryInput."""
    llm_instance = dialogflow.LLM(project_id="proj", agent_id="agent", language_code="fr")
    llm_instance._client = AsyncMock()
    llm_instance._client.detect_intent = AsyncMock(return_value=_make_mock_response(["Bonjour"]))

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Salut")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    async for _ in stream:
        pass

    call_args = llm_instance._client.detect_intent.call_args
    request = call_args.kwargs["request"]
    assert request.query_input.language_code == "fr"
