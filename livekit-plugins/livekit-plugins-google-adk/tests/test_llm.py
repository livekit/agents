"""Tests for Google ADK LLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import llm
from livekit.plugins.google_adk import LLM as GoogleADK


class TestGoogleADKLLM:
    """Test cases for GoogleADK LLM class."""

    def test_initialization(self):
        """Test LLM initialization with required parameters."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
        )

        assert adk_llm.model == "google-adk"
        assert adk_llm.provider == "google-adk"
        assert adk_llm._api_base_url == "http://localhost:8000"
        assert adk_llm._app_name == "test-app"
        assert adk_llm._user_id == "test-user"

    def test_initialization_with_custom_model(self):
        """Test LLM initialization with custom model name."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
            model="custom-adk-model",
        )

        assert adk_llm.model == "custom-adk-model"

    def test_initialization_with_session_id(self):
        """Test LLM initialization with existing session ID."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
            session_id="existing-session",
        )

        assert adk_llm._session_id == "existing-session"

    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test automatic session creation."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
        )

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"session_id": "test-session"})

        # client.post() should return a context manager (not a coroutine)
        mock_client = MagicMock()
        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_client.post.return_value = mock_post_ctx

        with patch.object(
            adk_llm, "_ensure_client_session", new=AsyncMock(return_value=mock_client)
        ):
            session_id = await adk_llm._create_session()
            assert session_id.startswith("session-")

    @pytest.mark.asyncio
    async def test_chat_streaming(self):
        """Test that chat method returns an LLMStream object."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
            session_id="test-session",
        )

        # Create mock chat context
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(
            role="user",
            content="Hello, how are you?",
        )

        # Verify chat returns an LLMStream
        stream = adk_llm.chat(chat_ctx=chat_ctx)
        assert stream is not None
        # Check that it's the right type
        from livekit.plugins.google_adk.llm_stream import LLMStream

        assert isinstance(stream, LLMStream)

    @pytest.mark.asyncio
    async def test_error_handling_session_creation(self):
        """Test error handling when session creation fails."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
        )

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(RuntimeError, match="Failed to create ADK session"):
                await adk_llm._create_session()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test resource cleanup."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
        )

        # Create a mock client session
        mock_session = AsyncMock()
        mock_session.closed = False
        adk_llm._client_session = mock_session

        await adk_llm.aclose()

        # Verify session was closed
        mock_session.close.assert_called_once()
        assert adk_llm._client_session is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
