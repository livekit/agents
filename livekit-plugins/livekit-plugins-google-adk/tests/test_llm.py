"""Tests for Google ADK LLM plugin."""

from unittest.mock import AsyncMock, patch

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

        # Mock the client session
        mock_client = AsyncMock()
        mock_client.post.return_value.__aenter__.return_value = mock_response

        with patch.object(adk_llm, "_ensure_client_session", return_value=mock_client):
            session_id = await adk_llm._create_session()
            assert session_id.startswith("session-")

    @pytest.mark.asyncio
    async def test_chat_streaming(self):
        """Test chat streaming with SSE response."""
        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="test-app",
            user_id="test-user",
            session_id="test-session",  # Provide session to skip creation
        )

        # Create mock chat context
        chat_ctx = llm.ChatContext()
        chat_ctx.messages.append(
            llm.ChatMessage(
                role=llm.ChatRole.USER,
                content=["Hello, how are you?"],
            )
        )

        # Mock SSE stream
        mock_stream_data = [
            b'data: {"content": {"parts": [{"text": "Hello"}]}, "partial": true}\n',
            b'data: {"content": {"parts": [{"text": "! How"}]}, "partial": true}\n',
            b'data: {"content": {"parts": [{"text": " can I"}]}, "partial": true}\n',
            b'data: {"content": {"parts": [{"text": " help?"}]}, "partial": true}\n',
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = AsyncMock()
        mock_response.content.__aiter__.return_value = iter(mock_stream_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            chunks = []
            async for chunk in adk_llm.chat(chat_ctx=chat_ctx):
                chunks.append(chunk)

            # Verify we received chunks
            assert len(chunks) == 4
            assert all(isinstance(chunk, llm.ChatChunk) for chunk in chunks)

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

        # Mock the client session
        mock_client = AsyncMock()
        mock_client.post.return_value.__aenter__.return_value = mock_response

        with patch.object(adk_llm, "_ensure_client_session", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Internal Server Error"):
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
        adk_llm._client_session = AsyncMock()
        adk_llm._client_session.closed = False

        await adk_llm.aclose()

        # Verify session was closed
        adk_llm._client_session.close.assert_called_once()
        assert adk_llm._client_session is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
