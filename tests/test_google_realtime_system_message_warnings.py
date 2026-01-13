"""
Tests for system message dropping warnings in Google Gemini Realtime API.

These tests verify that clear warnings are logged when system/developer messages
are dropped due to Gemini Realtime API's role constraints.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents import llm


@pytest.fixture
def mock_gemini_session():
    """Create a mock Gemini Realtime session for testing."""
    with patch("livekit.plugins.google.realtime.realtime_api.GenAIClient") as mock_client:
        # Mock the async context manager properly
        mock_async_session = AsyncMock()
        mock_async_session.receive = AsyncMock(return_value=AsyncMock())
        mock_async_session.__aiter__ = lambda self: self
        mock_async_session.__anext__ = AsyncMock(side_effect=asyncio.CancelledError())
        mock_async_session.send_client_content = AsyncMock()
        mock_async_session.send_tool_response = AsyncMock()
        mock_async_session.send_realtime_input = AsyncMock()
        mock_async_session.close = AsyncMock()

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_async_session)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        mock_client.return_value.aio.live.connect.return_value = mock_context_manager
        yield mock_async_session


@pytest.mark.asyncio
async def test_update_chat_ctx_warns_when_system_messages_present(mock_gemini_session, caplog):
    """Test that update_chat_ctx logs warning when system messages are dropped."""
    from livekit.plugins.google.realtime import RealtimeModel

    # Create a RealtimeModel instance
    model = RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key="test-key",
        voice="Puck",
    )

    # Create a session
    session = model.session()

    # Ensure session is cleaned up
    try:
        # Create chat context with system messages
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="You are a helpful assistant")
        chat_ctx.add_message(role="developer", content="Debug mode enabled")
        chat_ctx.add_message(role="user", content="Hello")
        chat_ctx.add_message(role="assistant", content="Hi there!")

        # Update chat context (should trigger warning)
        with caplog.at_level("WARNING"):
            await session.update_chat_ctx(chat_ctx)

        # Verify warning was logged
        assert len(caplog.records) == 1
        warning = caplog.records[0]

        # Check warning content
        assert "gemini-2.5-flash-native-audio-preview-12-2025" in warning.message
        assert "2" in warning.message  # 2 system messages dropped
        assert "system" in warning.message.lower() or "developer" in warning.message.lower()
        assert "user" in warning.message.lower() and "model" in warning.message.lower()
        assert "update_instructions" in warning.message
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_chat_ctx_no_warning_when_no_system_messages(mock_gemini_session, caplog):
    """Test that no warning is logged when chat context has no system messages."""
    from livekit.plugins.google.realtime import RealtimeModel

    model = RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key="test-key",
        voice="Puck",
    )

    session = model.session()

    try:
        # Create chat context with ONLY user/model messages
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content="Hello")
        chat_ctx.add_message(role="assistant", content="Hi there!")

        # Update chat context (should NOT trigger warning)
        with caplog.at_level("WARNING"):
            await session.update_chat_ctx(chat_ctx)

        # Verify no warning was logged
        assert len(caplog.records) == 0
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_chat_ctx_warns_every_time(mock_gemini_session, caplog):
    """Test that warning is logged EVERY time, not just once."""
    from livekit.plugins.google.realtime import RealtimeModel

    model = RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key="test-key",
        voice="Puck",
    )

    session = model.session()

    try:
        # Create chat context with system message
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="System instruction")
        chat_ctx.add_message(role="user", content="Hello")

        # Call update_chat_ctx multiple times
        with caplog.at_level("WARNING"):
            await session.update_chat_ctx(chat_ctx)
            await session.update_chat_ctx(chat_ctx)
            await session.update_chat_ctx(chat_ctx)

        # Verify warning was logged 3 times (every time)
        assert len(caplog.records) == 3
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_session_init_warns_when_system_messages_present(mock_gemini_session, caplog):
    """Test that session initialization logs warning when chat context has system messages."""
    from livekit.plugins.google.realtime import RealtimeModel

    model = RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key="test-key",
        voice="Puck",
    )

    # Create initial chat context with system messages
    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.add_message(role="system", content="System instruction")
    initial_chat_ctx.add_message(role="user", content="Hello")

    # Create session with initial context
    session = model.session()
    session._chat_ctx = initial_chat_ctx

    try:
        with caplog.at_level("WARNING"):
            # Trigger session initialization by starting the session
            # We'll need to wait a bit for the main task to start and process the chat context
            await asyncio.sleep(0.2)
            # Cancel the main task to avoid hanging
            session._main_atask.cancel()
            try:
                await session._main_atask
            except asyncio.CancelledError:
                pass

        # Verify warning was logged during initialization
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

        # Check that at least one warning mentions system messages
        warning_messages = " ".join(w.message for w in warnings)
        assert "system" in warning_messages.lower() or "developer" in warning_messages.lower()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_warning_counts_only_system_developer_roles(mock_gemini_session, caplog):
    """Test that warning only counts system and developer roles, not other types."""
    from livekit.plugins.google.realtime import RealtimeModel

    model = RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key="test-key",
        voice="Puck",
    )

    session = model.session()

    try:
        # Create chat context with mixed message types
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="System 1")
        chat_ctx.add_message(role="user", content="User 1")
        chat_ctx.add_message(role="assistant", content="Model 1")
        chat_ctx.add_message(role="developer", content="Developer 1")
        chat_ctx.add_message(role="user", content="User 2")
        chat_ctx.add_message(role="system", content="System 2")

        # Update chat context
        with caplog.at_level("WARNING"):
            await session.update_chat_ctx(chat_ctx)

        # Verify warning mentions exactly 3 messages (2 system + 1 developer)
        assert len(caplog.records) == 1
        assert "3" in caplog.records[0].message
    finally:
        await session.aclose()
