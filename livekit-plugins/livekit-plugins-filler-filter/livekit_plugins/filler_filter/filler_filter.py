"""
FillerFilterWrapper - Main STT wrapper for filtering filler interruptions
Wraps any STT provider and filters transcriptions based on agent state
"""
import logging
from typing import Optional
from livekit.agents import stt

from .config import FillerFilterConfig, get_config
from .state_tracker import AgentStateTracker
from .utils import should_filter


logger = logging.getLogger(__name__)


class FillerFilterWrapper(stt.STT):
    """
    STT wrapper that filters filler word interruptions based on agent speaking state.
    
    This wrapper intercepts the STT stream and filters out filler-only transcriptions
    when the agent is speaking, while allowing genuine interruptions through.
    """
    
    def __init__(
        self,
        stt: stt.STT,
        state_tracker: AgentStateTracker,
        config: Optional[FillerFilterConfig] = None
    ):
        """
        Initialize the filler filter wrapper
        
        Args:
            stt: Underlying STT provider (Deepgram, OpenAI, etc.)
            state_tracker: Agent state tracker instance
            config: Filter configuration (uses global config if None)
        """
        super().__init__(capabilities=stt.capabilities)
        
        self._stt = stt
        self._state_tracker = state_tracker
        self._config = config or get_config()
        
        # Statistics for debugging
        self._stats = {
            'total_events': 0,
            'filtered_events': 0,
            'passed_events': 0
        }
        
        logger.info(f"FillerFilterWrapper initialized with {type(stt).__name__}")
    
    def stream(
        self,
        *,
        language: Optional[str] = None
    ) -> "FillerFilterStream":
        """
        Create a new STT stream with filtering
        
        Args:
            language: Language code for transcription
        
        Returns:
            FillerFilterStream instance
        """
        base_stream = self._stt.stream(language=language)
        return FillerFilterStream(
            base_stream=base_stream,
            state_tracker=self._state_tracker,
            config=self._config,
            stats=self._stats
        )
    
    def get_stats(self):
        """Get filtering statistics"""
        return self._stats.copy()


class FillerFilterStream(stt.SpeechStream):
    """
    Filtered STT stream that processes and filters speech events
    """
    
    def __init__(
        self,
        base_stream: stt.SpeechStream,
        state_tracker: AgentStateTracker,
        config: FillerFilterConfig,
        stats: dict
    ):
        """
        Initialize the filtered stream
        
        Args:
            base_stream: Underlying STT stream
            state_tracker: Agent state tracker
            config: Filter configuration
            stats: Statistics dictionary to update
        """
        self._base_stream = base_stream
        self._state_tracker = state_tracker
        self._config = config
        self._stats = stats
    
    async def __anext__(self) -> stt.SpeechEvent:
        """
        Get next speech event, filtering as needed
        
        This is the core filtering logic that runs in the event loop.
        Uses async generator pattern for zero-copy streaming.
        """
        while True:
            # Get next event from base STT
            event = await self._base_stream.__anext__()
            
            self._stats['total_events'] += 1
            
            # Only filter final transcripts (interim transcripts pass through)
            if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT:
                self._stats['passed_events'] += 1
                return event
            
            # Check if filtering is enabled
            if not self._config.enabled:
                self._stats['passed_events'] += 1
                return event
            
            # Get the transcribed text
            if not event.alternatives or len(event.alternatives) == 0:
                self._stats['passed_events'] += 1
                return event
            
            alternative = event.alternatives
            text = alternative.text
            confidence = alternative.confidence
            
            # Get current agent speaking state
            is_agent_speaking = self._state_tracker.is_speaking_sync()
            
            # Determine if we should filter this event
            should_filter_event = should_filter(
                text=text,
                confidence=confidence,
                is_agent_speaking=is_agent_speaking,
                filler_words=self._config.ignored_words,
                confidence_threshold=self._config.confidence_threshold,
                debug=self._config.debug_mode
            )
            
            if should_filter_event:
                self._stats['filtered_events'] += 1
                logger.info(
                    f"[FILTERED] Agent speaking: {is_agent_speaking}, "
                    f"Confidence: {confidence:.2f}, Text: '{text}'"
                )
                # Continue to next event instead of returning this one
                continue
            else:
                self._stats['passed_events'] += 1
                if self._config.debug_mode:
                    logger.debug(
                        f"[PASSED] Agent speaking: {is_agent_speaking}, "
                        f"Confidence: {confidence:.2f}, Text: '{text}'"
                    )
                return event
    
    def push_frame(self, frame):
        """Push audio frame to the base stream"""
        return self._base_stream.push_frame(frame)
    
    async def aclose(self):
        """Close the stream"""
        await self._base_stream.aclose()
    
    async def flush(self):
        """Flush the stream"""
        if hasattr(self._base_stream, 'flush'):
            await self._base_stream.flush()
