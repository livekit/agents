"""Tests for the VADWrapper class."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from livekit import rtc
from livekit.agents import vad
from livekit.agents.voice import VADWrapper, VADWrapperConfig, FillerWordConfig, PhoneticConfig

class MockVADStream(vad.VADStream):
    """Mock VADStream for testing."""
    
    def __init__(self):
        super().__init__()
        self._pushed_frames = []
        self._flushed = False
    
    def push_frame(self, frame: rtc.AudioFrame):
        self._pushed_frames.append(frame)
    
    def flush(self):
        self._flushed = True

class MockVAD(vad.VAD):
    """Mock VAD for testing."""
    
    def __init__(self):
        super().__init__()
        self._streams = []
        self._stream_kwargs = None
    
    def stream(self, **kwargs):
        self._stream_kwargs = kwargs
        stream = MockVADStream()
        self._streams.append(stream)
        return stream

@pytest.mark.asyncio
async def test_vad_wrapper_basic():
    """Test basic VAD wrapper functionality."""
    # Create a mock VAD and wrapper
    mock_vad = MockVAD()
    config = VADWrapperConfig(
        filler_config=FillerWordConfig(
            filler_words=["um", "uh"],
            phonetic_config=PhoneticConfig(enabled=False)
        )
    )
    wrapper = VADWrapper(mock_vad, config)
    
    # Create a stream
    stream = wrapper.stream()
    assert isinstance(stream, MockVADStream)  # Should return the underlying stream
    
    # Test agent speaking state
    assert wrapper._is_agent_speaking is False
    wrapper.set_agent_speaking(True)
    assert wrapper._is_agent_speaking is True

@pytest.mark.asyncio
async def test_vad_wrapper_filler_filtering():
    """Test that filler words are filtered when agent is speaking."""
    # Create a mock VAD and wrapper
    mock_vad = MockVAD()
    config = VADWrapperConfig(
        filler_config=FillerWordConfig(
            filler_words=["um", "uh"],
            phonetic_config=PhoneticConfig(enabled=False)
        )
    )
    wrapper = VADWrapper(mock_vad, config)
    
    # Create a stream and get the wrapped stream
    stream = wrapper.stream()
    wrapped_stream = None
    for s in mock_vad._streams:
        if hasattr(s, '_stream'):  # This is our VADStreamWrapper
            wrapped_stream = s
            break
    
    assert wrapped_stream is not None
    
    # Create a mock event handler
    mock_handler = AsyncMock()
    wrapped_stream.on("inference_done", mock_handler)
    
    # Create test events
    filler_event = vad.VADEvent(
        start_time=0.0,
        end_time=0.5,
        speech=True,
        text="um",
        words=[vad.WordTimeSpan(word="um", start_time=0.0, end_time=0.5)]
    )
    
    speech_event = vad.VADEvent(
        start_time=0.5,
        end_time=1.0,
        speech=True,
        text="hello",
        words=[vad.WordTimeSpan(word="hello", start_time=0.5, end_time=1.0)]
    )
    
    # Test 1: Agent not speaking, all events should pass through
    wrapper.set_agent_speaking(False)
    wrapped_stream._on_inference_done(filler_event)
    wrapped_stream._on_inference_done(speech_event)
    
    # Both events should be emitted
    assert mock_handler.call_count == 2
    
    # Reset mock for next test
    mock_handler.reset_mock()
    
    # Test 2: Agent speaking, filler words should be filtered
    wrapper.set_agent_speaking(True)
    wrapped_stream._on_inference_done(filler_event)  # Should be filtered
    wrapped_stream._on_inference_done(speech_event)  # Should pass through
    
    # Only the non-filler event should be emitted
    assert mock_handler.call_count == 1
    assert mock_handler.call_args[0][0].text == "hello"

@pytest.mark.asyncio
async def test_vad_wrapper_update_config():
    """Test updating the VAD wrapper configuration."""
    # Create a mock VAD and wrapper
    mock_vad = MockVAD()
    wrapper = VADWrapper(mock_vad)
    
    # Update filler words
    new_words = ["like", "actually"]
    wrapper.update_filler_words(new_words)
    assert wrapper.config.filler_config.filler_words == new_words
    
    # Update phonetic config
    wrapper.update_phonetic_config(enabled=True, algorithm="metaphone")
    assert wrapper.config.filler_config.phonetic_config.enabled is True
    assert wrapper.config.filler_config.phonetic_config.algorithm == "metaphone"

@pytest.mark.asyncio
async def test_vad_wrapper_phonetic_matching():
    """Test phonetic matching in the VAD wrapper."""
    # Create a mock VAD and wrapper with phonetic matching enabled
    mock_vad = MockVAD()
    config = VADWrapperConfig(
        filler_config=FillerWordConfig(
            filler_words=["um"],
            phonetic_config=PhoneticConfig(
                enabled=True,
                algorithm="double_metaphone"
            )
        )
    )
    wrapper = VADWrapper(mock_vad, config)
    
    # Create a stream and get the wrapped stream
    stream = wrapper.stream()
    wrapped_stream = None
    for s in mock_vad._streams:
        if hasattr(s, '_stream'):  # This is our VADStreamWrapper
            wrapped_stream = s
            break
    
    assert wrapped_stream is not None
    
    # Create a mock event handler
    mock_handler = AsyncMock()
    wrapped_stream.on("inference_done", mock_handler)
    
    # Create test events with phonetic variations of "um"
    variations = ["umm", "uhm", "ummm", "uhhh"]
    
    # Set agent to speaking - these should all be filtered
    wrapper.set_agent_speaking(True)
    
    for i, variation in enumerate(variations):
        event = vad.VADEvent(
            start_time=float(i),
            end_time=float(i) + 0.5,
            speech=True,
            text=variation,
            words=[vad.WordTimeSpan(word=variation, start_time=float(i), end_time=float(i) + 0.5)]
        )
        wrapped_stream._on_inference_done(event)
    
    # None of the variations should have been emitted
    assert mock_handler.call_count == 0

if __name__ == "__main__":
    pytest.main([__file__])
