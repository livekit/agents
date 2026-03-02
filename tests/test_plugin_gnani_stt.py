"""Tests for Gnani STT plugin"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, stt
from livekit.agents.stt import SpeechData, SpeechEventType
from livekit.plugins.gnani import STT

from .conftest import TEST_CONNECT_OPTIONS


class TestGnaniSTT:
    """Test suite for Gnani STT plugin"""

    @pytest.fixture
    def mock_api_credentials(self, monkeypatch):
        """Set up mock API credentials"""
        monkeypatch.setenv("GNANI_API_KEY", "test-api-key")
        monkeypatch.setenv("GNANI_ORG_ID", "test-org-id")
        monkeypatch.setenv("GNANI_USER_ID", "test-user-id")

    @pytest.fixture
    def audio_buffer(self):
        """Create a simple audio buffer for testing"""
        # Create a simple audio frame with 1 second of silence
        sample_rate = 16000
        num_channels = 1
        samples_per_channel = sample_rate  # 1 second
        audio_data = b"\x00" * (samples_per_channel * num_channels * 2)  # 16-bit samples

        frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )
        return [frame]

    async def test_stt_initialization(self, mock_api_credentials):
        """Test STT initialization with credentials"""
        stt_instance = STT(language="en-IN")
        assert stt_instance.model == "gnani-stt-v3"
        assert stt_instance.provider == "Gnani"
        assert stt_instance._opts.language == "en-IN"

    async def test_stt_initialization_with_params(self):
        """Test STT initialization with explicit parameters"""
        stt_instance = STT(
            language="hi-IN",
            api_key="custom-api-key",
            organization_id="custom-org-id",
            user_id="custom-user-id",
        )
        assert stt_instance._opts.language == "hi-IN"
        assert stt_instance._api_key == "custom-api-key"
        assert stt_instance._organization_id == "custom-org-id"
        assert stt_instance._user_id == "custom-user-id"

    async def test_stt_missing_credentials(self, monkeypatch):
        """Test that STT raises error when credentials are missing"""
        # Clear all environment variables
        monkeypatch.delenv("GNANI_API_KEY", raising=False)
        monkeypatch.delenv("GNANI_ORG_ID", raising=False)
        monkeypatch.delenv("GNANI_USER_ID", raising=False)

        with pytest.raises(ValueError, match="Gnani API key is required"):
            STT(language="en-IN")

    async def test_stt_missing_org_id(self, monkeypatch):
        """Test that STT raises error when organization ID is missing"""
        # Clear all environment variables first
        monkeypatch.delenv("GNANI_API_KEY", raising=False)
        monkeypatch.delenv("GNANI_ORG_ID", raising=False)
        monkeypatch.delenv("GNANI_USER_ID", raising=False)

        # Set only API key
        monkeypatch.setenv("GNANI_API_KEY", "test-api-key")
        with pytest.raises(ValueError, match="Gnani Organization ID is required"):
            STT(language="en-IN")

    async def test_stt_missing_user_id(self, monkeypatch):
        """Test that STT raises error when user ID is missing"""
        # Clear all environment variables first
        monkeypatch.delenv("GNANI_API_KEY", raising=False)
        monkeypatch.delenv("GNANI_ORG_ID", raising=False)
        monkeypatch.delenv("GNANI_USER_ID", raising=False)

        # Set only API key and org ID
        monkeypatch.setenv("GNANI_API_KEY", "test-api-key")
        monkeypatch.setenv("GNANI_ORG_ID", "test-org-id")
        with pytest.raises(ValueError, match="Gnani User ID is required"):
            STT(language="en-IN")

    async def test_update_options(self, mock_api_credentials):
        """Test updating STT options"""
        stt_instance = STT(language="en-IN")
        assert stt_instance._opts.language == "en-IN"

        stt_instance.update_options(language="hi-IN")
        assert stt_instance._opts.language == "hi-IN"

    async def test_recognize_success(self, mock_api_credentials, audio_buffer, job_process):
        """Test successful recognition"""
        stt_instance = STT(language="en-IN")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True, "transcript": "Hello world"})

        with patch.object(aiohttp.ClientSession, "post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            event = await stt_instance.recognize(buffer=audio_buffer)

            assert isinstance(event, stt.SpeechEvent)
            assert event.type == SpeechEventType.FINAL_TRANSCRIPT
            assert len(event.alternatives) == 1
            assert event.alternatives[0].text == "Hello world"
            assert event.alternatives[0].language == "en-IN"
            assert event.alternatives[0].confidence == 1.0

    async def test_recognize_with_language_override(
        self, mock_api_credentials, audio_buffer, job_process
    ):
        """Test recognition with language override"""
        stt_instance = STT(language="en-IN")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True, "transcript": "नमस्ते"})

        with patch.object(aiohttp.ClientSession, "post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            event = await stt_instance.recognize(buffer=audio_buffer, language="hi-IN")

            assert event.alternatives[0].text == "नमस्ते"
            assert event.alternatives[0].language == "hi-IN"

    async def test_recognize_api_error(self, mock_api_credentials, audio_buffer, job_process):
        """Test handling of API errors"""
        stt_instance = STT(language="en-IN")

        # Mock the API response with error
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")

        with patch.object(aiohttp.ClientSession, "post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(APIStatusError):
                await stt_instance.recognize(buffer=audio_buffer, conn_options=TEST_CONNECT_OPTIONS)

    async def test_recognize_connection_error(
        self, mock_api_credentials, audio_buffer, job_process
    ):
        """Test handling of connection errors"""
        stt_instance = STT(language="en-IN")

        with patch.object(aiohttp.ClientSession, "post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Connection failed")

            with pytest.raises(APIConnectionError):
                await stt_instance.recognize(buffer=audio_buffer)

    async def test_recognize_unsuccessful_response(
        self, mock_api_credentials, audio_buffer, job_process
    ):
        """Test handling of unsuccessful API response"""
        stt_instance = STT(language="en-IN")

        # Mock the API response with success=False
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"success": False, "error": "Transcription failed"}
        )

        with patch.object(aiohttp.ClientSession, "post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(APIConnectionError, match=r"Transcription failed"):
                await stt_instance.recognize(buffer=audio_buffer, conn_options=TEST_CONNECT_OPTIONS)

    async def test_capabilities(self, mock_api_credentials):
        """Test STT capabilities"""
        stt_instance = STT(language="en-IN")
        caps = stt_instance.capabilities

        assert caps.streaming is False
        assert caps.interim_results is False

    async def test_create_speech_event(self, mock_api_credentials):
        """Test speech event creation"""
        stt_instance = STT(language="en-IN")

        event = stt_instance._create_speech_event(
            text="Test transcript",
            language="hi-IN",
            request_id="test-request-123",
        )

        assert isinstance(event, stt.SpeechEvent)
        assert event.type == SpeechEventType.FINAL_TRANSCRIPT
        assert event.request_id == "test-request-123"
        assert len(event.alternatives) == 1

        alt = event.alternatives[0]
        assert isinstance(alt, SpeechData)
        assert alt.text == "Test transcript"
        assert alt.language == "hi-IN"
        assert alt.confidence == 1.0

    async def test_supported_languages(self, mock_api_credentials):
        """Test that STT accepts various supported languages"""
        languages = [
            "en-IN",
            "hi-IN",
            "gu-IN",
            "ta-IN",
            "kn-IN",
            "te-IN",
            "mr-IN",
            "bn-IN",
            "ml-IN",
            "pa-IN",
            "en-IN,hi-IN",
        ]

        for lang in languages:
            stt_instance = STT(language=lang)
            assert stt_instance._opts.language == lang

    async def test_custom_base_url(self, mock_api_credentials):
        """Test STT with custom base URL"""
        custom_url = "https://custom.api.vachana.ai/stt/v3"
        stt_instance = STT(language="en-IN", base_url=custom_url)
        assert stt_instance._base_url == custom_url

    async def test_http_session_reuse(self, mock_api_credentials):
        """Test that HTTP session can be provided"""
        async with aiohttp.ClientSession() as session:
            stt_instance = STT(language="en-IN", http_session=session)
            assert stt_instance._session == session
