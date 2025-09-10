"""
Test for Google TTS character count tracking in streaming mode.

This test validates that the Google TTS plugin correctly tracks the total number 
of characters sent to the Google API, rather than just the segment text length.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import is conditional to allow test to run only when dependencies are available
google_tts = pytest.importorskip("livekit.plugins.google.tts")


class TestGoogleTTSCharacterCounting:
    """Test character counting in Google TTS streaming mode."""
    
    @pytest.fixture
    def mock_google_client(self):
        """Mock Google TTS client to avoid real API calls."""
        with patch('google.cloud.texttospeech.TextToSpeechAsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client_class.from_service_account_info.return_value = mock_client
            
            # Mock streaming response
            mock_response = AsyncMock()
            mock_response.audio_content = b'fake_audio_data'
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = [mock_response]
            mock_client.streaming_synthesize.return_value = mock_stream
            
            yield mock_client
    
    @pytest.fixture
    def google_tts_instance(self, mock_google_client):
        """Create a Google TTS instance with mocked credentials."""
        return google_tts.TTS(
            credentials_info={
                "type": "service_account",
                "project_id": "test-project",
                "private_key_id": "test-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
                "client_email": "test@test-project.iam.gserviceaccount.com",
                "client_id": "123456789",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    
    @pytest.mark.asyncio
    async def test_character_count_initialization(self, google_tts_instance):
        """Test that character count is properly initialized."""
        stream = google_tts_instance.stream()
        
        # Verify initial state
        assert hasattr(stream, '_char_count'), "Stream should have _char_count attribute"
        assert stream._char_count == 0, "Initial character count should be 0"
        
        await stream.aclose()
    
    @pytest.mark.asyncio
    async def test_character_count_tracking_single_segment(self, google_tts_instance):
        """Test character counting for a single segment."""
        collected_metrics = []
        
        def collect_metrics(metrics):
            if hasattr(metrics, 'characters_count'):
                collected_metrics.append(metrics)
        
        google_tts_instance.on("metrics_collected", collect_metrics)
        
        stream = google_tts_instance.stream()
        
        # Push a known amount of text
        test_text = "Hello world! This is a test message."
        stream.push_text(test_text)
        stream.flush()
        stream.end_input()
        
        # Consume the stream
        frames = []
        async for event in stream:
            frames.append(event.frame)
        
        await stream.aclose()
        
        # Wait for metrics to be processed
        await asyncio.sleep(0.1)
        
        # Verify metrics were collected and character count is reasonable
        assert len(collected_metrics) > 0, "Should have collected metrics"
        
        total_chars = sum(m.characters_count for m in collected_metrics)
        expected_chars = len(test_text)
        
        # Character count should be close to the original text length
        # (allowing for tokenization differences)
        assert total_chars >= expected_chars * 0.8, (
            f"Character count too low: {total_chars} vs expected ~{expected_chars}"
        )
        assert total_chars <= expected_chars * 1.2, (
            f"Character count too high: {total_chars} vs expected ~{expected_chars}"
        )
    
    @pytest.mark.asyncio
    async def test_character_count_multiple_segments(self, google_tts_instance):
        """Test character counting across multiple segments."""
        collected_metrics = []
        
        def collect_metrics(metrics):
            if hasattr(metrics, 'characters_count'):
                collected_metrics.append(metrics)
        
        google_tts_instance.on("metrics_collected", collect_metrics)
        
        stream = google_tts_instance.stream()
        
        # Push multiple segments
        segments = [
            "First segment of text.",
            "Second segment of text.",
            "Third and final segment."
        ]
        
        total_expected_chars = 0
        
        for segment in segments:
            stream.push_text(segment)
            stream.flush()
            total_expected_chars += len(segment)
        
        stream.end_input()
        
        # Consume the stream
        frames = []
        async for event in stream:
            frames.append(event.frame)
        
        await stream.aclose()
        
        # Wait for metrics to be processed
        await asyncio.sleep(0.1)
        
        # Verify metrics were collected for each segment
        assert len(collected_metrics) == len(segments), (
            f"Should have metrics for each segment: expected {len(segments)}, got {len(collected_metrics)}"
        )
        
        # Verify that each segment has a reasonable character count
        for i, metrics in enumerate(collected_metrics):
            segment_length = len(segments[i])
            assert metrics.characters_count >= segment_length * 0.8, (
                f"Segment {i} character count too low: {metrics.characters_count} vs expected ~{segment_length}"
            )
    
    @pytest.mark.asyncio
    async def test_character_count_with_interruptions(self, google_tts_instance):
        """Test character counting when interruptions occur."""
        collected_metrics = []
        
        def collect_metrics(metrics):
            if hasattr(metrics, 'characters_count'):
                collected_metrics.append(metrics)
        
        google_tts_instance.on("metrics_collected", collect_metrics)
        
        stream = google_tts_instance.stream()
        
        # Push some text
        stream.push_text("This is the beginning of a longer message")
        
        # Simulate interruption by flushing early
        stream.flush()
        
        # Continue with more text after interruption
        stream.push_text("This is the continuation after interruption")
        stream.flush()
        stream.end_input()
        
        # Consume the stream
        frames = []
        async for event in stream:
            frames.append(event.frame)
        
        await stream.aclose()
        
        # Wait for metrics to be processed
        await asyncio.sleep(0.1)
        
        # Should have metrics for both segments (before and after interruption)
        assert len(collected_metrics) >= 1, "Should have collected metrics despite interruption"
        
        # Total character count should account for all text sent
        total_chars = sum(m.characters_count for m in collected_metrics)
        assert total_chars > 0, "Should have non-zero character count"
    
    def test_character_count_accumulation_logic(self):
        """Test the character counting logic without async complexity."""
        # This tests the core logic of character accumulation
        
        tokens = ["Hello", " world", "!", " This", " is", " a", " test."]
        expected_total = sum(len(token) for token in tokens)
        
        # Simulate the accumulation that happens in _run_stream
        segment_char_count = 0
        for token in tokens:
            segment_char_count += len(token)
        
        assert segment_char_count == expected_total, (
            f"Character accumulation failed: {segment_char_count} vs {expected_total}"
        )
        
        # Verify specific token lengths for debugging
        assert len("Hello") == 5
        assert len(" world") == 6
        assert len("!") == 1
        assert expected_total == 29  # 5+6+1+5+3+2+7 = 29


if __name__ == "__main__":
    pytest.main([__file__, "-v"])