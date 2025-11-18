"""VAD wrapper for filler word detection."""
from typing import Optional, Dict, Any, Callable, List
import asyncio
import logging
from dataclasses import dataclass, field

from livekit import rtc
from livekit.agents import vad

from .filler_detector import FillerWordDetector, FillerWordConfig

logger = logging.getLogger(__name__)

@dataclass
class VADWrapperConfig:
    """Configuration for the VAD wrapper.
    
    Attributes:
        filler_config: Configuration for filler word detection
        log_filtered: Whether to log filtered filler words
    """
    filler_config: FillerWordConfig = field(default_factory=FillerWordConfig)
    log_filtered: bool = True

class VADStreamWrapper(vad.VADStream):
    """Wraps a VAD stream to filter filler words."""
    
    def __init__(
        self,
        stream: vad.VADStream,
        detector: FillerWordDetector,
        is_agent_speaking: Callable[[], bool],
        log_filtered: bool = True
    ):
        """Initialize the VAD stream wrapper.
        
        Args:
            stream: The underlying VAD stream
            detector: Filler word detector instance
            is_agent_speaking: Callback to check if agent is speaking
            log_filtered: Whether to log filtered filler words
        """
        super().__init__()
        self._stream = stream
        self._detector = detector
        self._is_agent_speaking = is_agent_speaking
        self._log_filtered = log_filtered
        
        # Forward events from the underlying stream
        self._stream.on("start_of_speech", self._on_start_of_speech)
        self._stream.on("inference_done", self._on_inference_done)
        self._stream.on("end_of_speech", self._on_end_of_speech)
    
    def push_frame(self, frame: rtc.AudioFrame):
        """Forward audio frames to the underlying stream.
        
        Args:
            frame: Audio frame to process
        """
        self._stream.push_frame(frame)
    
    def flush(self):
        """Flush the underlying stream."""
        self._stream.flush()
    
    def _on_start_of_speech(self, event: vad.VADEvent):
        """Handle start of speech events.
        
        Args:
            event: VAD event data
        """
        self.emit("start_of_speech", event)
    
    def _on_inference_done(self, event: vad.VADEvent):
        """Handle inference done events with filler word filtering.
        
        Args:
            event: VAD event data
        """
        # If agent is not speaking, just forward all events
        if not self._is_agent_speaking():
            self.emit("inference_done", event)
            return
            
        # If agent is speaking, check for filler words
        if not event.speech or not event.text or not event.words:
            self.emit("inference_done", event)
            return
            
        # Check if this is just filler words
        all_fillers = all(self._detector.is_filler(word.word) for word in event.words)
        
        if all_fillers:
            if self._log_filtered:
                logger.debug(f"Filtered filler words: {event.text}")
            # Don't emit the event if it's just filler words while agent is speaking
            return
            
        # Otherwise, forward the event
        self.emit("inference_done", event)
    
    def _on_end_of_speech(self, event: vad.VADEvent):
        """Handle end of speech events.
        
        Args:
            event: VAD event data
        """
        self.emit("end_of_speech", event)

class VADWrapper(vad.VAD):
    """Wraps a VAD instance to add filler word awareness."""
    
    def __init__(self, vad_instance: vad.VAD, config: Optional[VADWrapperConfig] = None):
        """Initialize the VAD wrapper.
        
        Args:
            vad_instance: The underlying VAD instance to wrap
            config: Configuration for the wrapper
        """
        super().__init__()
        self._vad = vad_instance
        self.config = config or VADWrapperConfig()
        self._detector = FillerWordDetector(self.config.filler_config)
        self._is_agent_speaking = False
        
        # Forward events from the underlying VAD
        self._vad.on("metrics_collected", self._on_metrics)
    
    def set_agent_speaking(self, is_speaking: bool):
        """Update the agent's speaking state.
        
        Args:
            is_speaking: Whether the agent is currently speaking
        """
        self._is_agent_speaking = is_speaking
        logger.debug(f"Agent speaking state updated: {is_speaking}")
    
    def update_filler_words(self, words: List[str]):
        """Update the list of filler words at runtime.
        
        Args:
            words: New list of filler words
        """
        self.config.filler_config.filler_words = words or []
        self._detector.update_config(self.config.filler_config)
        logger.info(f"Updated filler words: {words}")
    
    def add_phonetic_mapping(self, word: str, phonetic_forms: List[str]):
        """Add custom phonetic mappings for a word.
        
        Args:
            word: The word to add mappings for
            phonetic_forms: List of phonetic forms for the word
        """
        self._detector.add_filler_word(word)
        for form in phonetic_forms:
            self._detector.add_phonetic_mapping(word, form)
        logger.info(f"Added phonetic mapping: {word} -> {phonetic_forms}")
    
    def update_phonetic_config(self, **kwargs):
        """Update phonetic matching configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self._detector.config.phonetic_config, key):
                setattr(self._detector.config.phonetic_config, key, value)
            else:
                logger.warning(f"Unknown phonetic config parameter: {key}")
        
        self._detector.update_config(self._detector.config)
        logger.info(f"Updated phonetic config: {kwargs}")
    
    def stream(
        self,
        *,
        min_speaking_time: float = 0.5,
        min_silence_time: float = 0.5,
        padding_duration: float = 0.1,
        min_speech_segment_duration: float = 0.5,
        max_speech_segment_duration: float = 10.0,
        **kwargs
    ) -> VADStreamWrapper:
        """Create a new VAD stream with filler word filtering.
        
        Args:
            min_speaking_time: Minimum speaking time in seconds
            min_silence_time: Minimum silence time in seconds
            padding_duration: Padding duration in seconds
            min_speech_segment_duration: Minimum speech segment duration in seconds
            max_speech_segment_duration: Maximum speech segment duration in seconds
            **kwargs: Additional arguments to pass to the underlying VAD stream
            
        Returns:
            A new VADStreamWrapper instance
        """
        stream = self._vad.stream(
            min_speaking_time=min_speaking_time,
            min_silence_time=min_silence_time,
            padding_duration=padding_duration,
            min_speech_segment_duration=min_speech_segment_duration,
            max_speech_segment_duration=max_speech_segment_duration,
            **kwargs
        )
        
        return VADStreamWrapper(
            stream=stream,
            detector=self._detector,
            is_agent_speaking=lambda: self._is_agent_speaking,
            log_filtered=self.config.log_filtered
        )
    
    def _on_metrics(self, metrics: Dict[str, Any]):
        """Forward metrics from the underlying VAD.
        
        Args:
            metrics: VAD metrics
        """
        self.emit("metrics_collected", metrics)
