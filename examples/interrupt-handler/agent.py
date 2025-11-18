"""
Example LiveKit Agent with Intelligent Interruption Handling
"""

import asyncio
import logging
from typing import Optional
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.plugins import openai, deepgram, silero

# Import our custom handler
from .interrupt_handler import IntelligentInterruptionHandler

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Voice assistant with intelligent interruption handling"""
    
    def __init__(
        self,
        context: JobContext,
        ignored_words: Optional[list] = None
    ):
        self.ctx = context
        
        # Initialize interruption handler
        self.interrupt_handler = IntelligentInterruptionHandler(
            ignored_words=ignored_words,
            confidence_threshold=0.6,
            enable_logging=True
        )
        
        # Agent state
        self.agent_output = None
        self.assistant = None
        
    async def start(self):
        """Start the voice assistant"""
        # Connect to room
        await self.ctx.connect()
        
        # Get the participant
        participant = await self.ctx.wait_for_participant()
        
        logger.info(f"Connected to participant: {participant.identity}")
        
        # Setup STT (Speech-to-Text)
        stt = deepgram.STT()
        
        # Setup TTS (Text-to-Speech)
        tts = openai.TTS()
        
        # Setup LLM
        llm = openai.LLM()
        
        # Create assistant with event handlers
        self.assistant = agents.VoiceAssistant(
            vad=silero.VAD.load(),
            stt=stt,
            llm=llm,
            tts=tts,
            chat_ctx=self._create_chat_context()
        )
        
        # Register our custom event handlers
        self._register_event_handlers()
        
        # Start the assistant
        self.assistant.start(self.ctx.room, participant)
        
        logger.info("Voice assistant started with intelligent interruption handling")
        
        # Keep running
        await asyncio.Future()
    
    def _create_chat_context(self):
        """Create initial chat context"""
        return agents.ChatContext(
            messages=[
                agents.ChatMessage(
                    role="system",
                    content=(
                        "You are a helpful voice assistant. Keep responses "
                        "concise and conversational."
                    )
                )
            ]
        )
    
    def _register_event_handlers(self):
        """Register event handlers for interruption logic"""
        
        # Track when agent starts speaking
        @self.assistant.on("agent_speech_started")
        def on_agent_speech_started():
            logger.debug("Agent started speaking")
            self.interrupt_handler.set_agent_speaking(True)
        
        # Track when agent stops speaking
        @self.assistant.on("agent_speech_stopped")
        def on_agent_speech_stopped():
            logger.debug("Agent stopped speaking")
            self.interrupt_handler.set_agent_speaking(False)
        
        # Intercept user speech events
        @self.assistant.on("user_speech_committed")
        async def on_user_speech(msg: agents.STTResult):
            await self._handle_user_speech(msg)
        
        # Handle interruptions during agent speech
        @self.assistant.on("user_started_speaking")
        async def on_user_started_speaking():
            await self._handle_potential_interruption()
    
    async def _handle_user_speech(self, stt_result):
        """Process user speech through our intelligent handler."""
        transcript = stt_result.text
        confidence = getattr(stt_result, 'confidence', 1.0)
        
        # Check if we should process this interruption
        should_process, event = await self.interrupt_handler.should_process_interruption(
            transcript=transcript,
            confidence=confidence
        )
        
        if not should_process:
            logger.info(f"Filtered out filler: '{transcript}'")
            # Don't pass to the assistant - it will continue speaking
            return
        
        # It's a genuine interruption - let it through
        logger.info(f"Processing genuine speech: '{transcript}'")
        # The assistant will handle this normally
    
    async def _handle_potential_interruption(self):
        """Called when user starts speaking (VAD triggered)."""
        if self.interrupt_handler.is_agent_speaking():
            logger.debug("User started speaking while agent is talking")
            # The transcription will be evaluated in _handle_user_speech
    
    def get_stats(self):
        """Get interruption handler statistics"""
        return self.interrupt_handler.get_stats()


# Worker entry point
async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit worker"""
    
    # Custom ignored words (optional - uses defaults if not specified)
    ignored_words = [
        'uh', 'uhh', 'um', 'umm', 'hmm', 'hmmm',
        'err', 'ah', 'ahh', 'haan', 'mhmm',
        'uh-huh', 'mm-hmm'
    ]
    
    # Create and start assistant
    assistant = VoiceAssistant(
        context=ctx,
        ignored_words=ignored_words
    )
    
    try:
        await assistant.start()
    except Exception as e:
        logger.error(f"Error in voice assistant: {e}", exc_info=True)
        raise
    finally:
        # Log statistics on shutdown
        stats = assistant.get_stats()
        logger.info(f"Session statistics: {stats}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the worker
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )