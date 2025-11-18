"""
Example integration of IntelligentInterruptionHandler with LiveKit Agent
"""

import asyncio
import logging
from typing import Optional

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

from livekit.agents.interruption import (
    IntelligentInterruptionHandler,
    LiveKitInterruptionWrapper
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedVoiceAssistant:
    """Voice assistant with intelligent interruption handling"""
    
    def __init__(
        self,
        vad: silero.VAD,
        stt_provider: stt.STT,
        llm_provider: llm.LLM,
        tts_provider: tts.TTS,
        ignored_words: Optional[list] = None,
        confidence_threshold: float = 0.6
    ):
        # Initialize interruption handler
        self.interruption_handler = IntelligentInterruptionHandler(
            ignored_words=ignored_words,
            confidence_threshold=confidence_threshold,
            log_all_events=True,
            allow_runtime_updates=True
        )
        
        # Create the voice assistant
        self.assistant = VoiceAssistant(
            vad=vad,
            stt=stt_provider,
            llm=llm_provider,
            tts=tts_provider,
            # We'll override interruption handling
        )
        
        # Wrap the interruption logic
        self.wrapper = LiveKitInterruptionWrapper(
            handler=self.interruption_handler,
            original_interrupt_callback=None  # Will be set dynamically
        )
        
        # Hook into assistant events
        self._setup_event_hooks()
    
    def _setup_event_hooks(self):
        """Set up event hooks to track agent speaking state"""
        
        @self.assistant.on("agent_started_speaking")
        def on_agent_start():
            asyncio.create_task(self.wrapper.on_agent_speech_start())
        
        @self.assistant.on("agent_stopped_speaking")
        def on_agent_stop():
            asyncio.create_task(self.wrapper.on_agent_speech_end())
        
        # Override the STT event handling
        original_on_speech = self.assistant._on_speech_event
        
        async def enhanced_on_speech(event: stt.SpeechEvent):
            """Enhanced speech event handler with intelligent filtering"""
            # Process through our intelligent handler first
            if not hasattr(event, 'alternatives') or not event.alternatives:
                return await original_on_speech(event)
            
            alternative = event.alternatives[0]
            text = alternative.text
            confidence = getattr(alternative, 'confidence', 1.0)
            
            # Check if this should interrupt
            should_interrupt = await self.interruption_handler.process_transcript(
                text=text,
                confidence=confidence,
                is_final=event.is_final
            )
            
            # Only process if it's a valid interruption or agent isn't speaking
            if should_interrupt or not self.interruption_handler._agent_speaking:
                await original_on_speech(event)
        
        # Replace the speech event handler
        self.assistant._on_speech_event = enhanced_on_speech
    
    async def start(self, room: rtc.Room):
        """Start the assistant"""
        await self.assistant.start(room)
    
    async def say(self, message: str, allow_interruptions: bool = True):
        """Have the assistant speak"""
        await self.assistant.say(message, allow_interruptions=allow_interruptions)
    
    def get_statistics(self) -> dict:
        """Get interruption statistics"""
        return self.interruption_handler.get_statistics()


async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    
    # Initialize components
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a helpful voice assistant. Be conversational and natural. "
            "Keep responses concise unless asked for detail."
        ),
    )
    
    # Configure ignored words (can be loaded from env or config)
    ignored_words = [
        'uh', 'um', 'umm', 'hmm', 'hm', 'mm', 'mhmm',
        'haan', 'han', 'ha', 'ah', 'eh', 'er', 'yeah'
    ]
    
    # Create enhanced assistant
    assistant = EnhancedVoiceAssistant(
        vad=silero.VAD.load(),
        stt_provider=deepgram.STT(
            language="en-US",
            detect_language=False,
            interim_results=True,
        ),
        llm_provider=openai.LLM(model="gpt-4"),
        tts_provider=openai.TTS(voice="alloy"),
        ignored_words=ignored_words,
        confidence_threshold=0.6
    )
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Start the assistant
    assistant.assistant.start(ctx.room, participant=ctx.room.remote_participants.values().__iter__().__next__())
    
    # Greet the user
    await assistant.say("Hello! I'm your voice assistant. How can I help you today?")
    
    # Print statistics periodically
    async def print_stats():
        while True:
            await asyncio.sleep(60)  # Every minute
            stats = assistant.get_statistics()
            logger.info(f"Interruption Statistics: {stats}")
    
    stats_task = asyncio.create_task(print_stats())
    
    try:
        # Keep the agent running
        await asyncio.sleep(float('inf'))
    finally:
        stats_task.cancel()


if __name__ == "__main__":
    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )