"""Tests for ElevenLabs eleven_v3 TTS model support"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.plugins import elevenlabs


class TestElevenLabsV3Support:
    """Test that eleven_v3 model uses HTTP streaming instead of WebSocket"""

    @pytest.fixture
    def mock_http_session(self):
        """Mock HTTP session for testing"""
        session = MagicMock()
        session.post = AsyncMock()
        return session

    def test_eleven_v3_in_non_websocket_models(self):
        """Verify eleven_v3 is marked as a non-WebSocket model"""
        from livekit.plugins.elevenlabs.tts import NON_WEBSOCKET_MODELS

        assert "eleven_v3" in NON_WEBSOCKET_MODELS

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_returns_http_stream_for_v3(self):
        """Test that stream() returns HTTPSynthesizeStream for eleven_v3"""
        from livekit.plugins.elevenlabs.tts import HTTPSynthesizeStream

        tts = elevenlabs.TTS(model="eleven_v3")
        stream = tts.stream()

        assert isinstance(stream, HTTPSynthesizeStream)
        assert tts.model == "eleven_v3"

        await stream.aclose()
        await tts.aclose()

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_returns_websocket_stream_for_other_models(self):
        """Test that stream() returns SynthesizeStream for non-v3 models"""
        from livekit.plugins.elevenlabs.tts import SynthesizeStream

        tts = elevenlabs.TTS(model="eleven_turbo_v2_5")
        stream = tts.stream()

        assert isinstance(stream, SynthesizeStream)
        assert tts.model == "eleven_turbo_v2_5"

        await stream.aclose()
        await tts.aclose()

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_model_update_to_v3_changes_stream_type(self):
        """Test that updating to eleven_v3 changes the stream type"""
        from livekit.plugins.elevenlabs.tts import HTTPSynthesizeStream, SynthesizeStream

        tts = elevenlabs.TTS(model="eleven_turbo_v2_5")
        stream1 = tts.stream()
        assert isinstance(stream1, SynthesizeStream)
        await stream1.aclose()

        tts.update_options(model="eleven_v3")
        stream2 = tts.stream()
        assert isinstance(stream2, HTTPSynthesizeStream)
        await stream2.aclose()
        await tts.aclose()

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_synthesize_works_for_v3(self):
        """Test that synthesize() still works for eleven_v3"""
        from livekit.plugins.elevenlabs.tts import ChunkedStream

        tts = elevenlabs.TTS(model="eleven_v3")
        stream = tts.synthesize("Hello world")

        assert isinstance(stream, ChunkedStream)
        await stream.aclose()
        await tts.aclose()

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_all_tts_models_are_supported(self):
        """Test that all model types can be instantiated"""
        models = [
            "eleven_monolingual_v1",
            "eleven_multilingual_v1",
            "eleven_multilingual_v2",
            "eleven_turbo_v2",
            "eleven_turbo_v2_5",
            "eleven_flash_v2_5",
            "eleven_flash_v2",
            "eleven_v3",
        ]

        for model in models:
            tts = elevenlabs.TTS(model=model)
            assert tts.model == model
            stream = tts.stream()
            assert stream is not None
            await stream.aclose()
            await tts.aclose()

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    def test_alignment_capability_for_v3(self):
        """Test that eleven_v3 doesn't support alignment yet (uses HTTP streaming)"""
        # Alignment not yet supported for HTTP streaming models
        tts_with_alignment = elevenlabs.TTS(model="eleven_v3", sync_alignment=True)
        assert tts_with_alignment.capabilities.aligned_transcript is False

        tts_without_alignment = elevenlabs.TTS(model="eleven_v3", sync_alignment=False)
        assert tts_without_alignment.capabilities.aligned_transcript is False

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"})
    def test_with_timestamps_url_generation(self):
        """Test that with-timestamps endpoint is used when sync_alignment is enabled"""
        from livekit.plugins.elevenlabs.tts import _synthesize_url, _TTSOptions

        opts = _TTSOptions(
            api_key="test",
            voice_id="voice123",
            model="eleven_v3",
            base_url="https://api.elevenlabs.io/v1",
            encoding="mp3_22050_32",
            sample_rate=22050,
            streaming_latency=0,
            word_tokenizer=None,
            chunk_length_schedule=[],
            enable_ssml_parsing=False,
            enable_logging=True,
            inactivity_timeout=180,
            sync_alignment=True,
            apply_text_normalization="auto",
            preferred_alignment="normalized",
            auto_mode=True,
            voice_settings=None,
            language=None,
            pronunciation_dictionary_locators=None,
        )

        url_with_timestamps = _synthesize_url(opts, with_timestamps=True)
        assert "stream/with-timestamps" in url_with_timestamps

        url_without_timestamps = _synthesize_url(opts, with_timestamps=False)
        assert "stream/with-timestamps" not in url_without_timestamps
        assert "/stream?" in url_without_timestamps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
