# examples/my_intelligent_agent.py

from livekit.agents import Agent, AgentSession, JobContext, cli
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit.agents.utils import function_tool # Example tool import

# Import modules created in Step 2
from .config import IGNORED_WORDS, INTERRUPTION_COMMANDS, LOW_CONFIDENCE_THRESHOLD
from .IntelligentInterruptHandler import IntelligentInterruptHandler

async def entrypoint(ctx: JobContext):
    """The starting point for an interactive session."""
    
    # 1. Connect to the LiveKit Room
    await ctx.connect()
    room = ctx.room
    
    # 2. Initialize Components and Handler
    interrupt_handler = IntelligentInterruptHandler(
        ignored_words=IGNORED_WORDS,
        interruption_commands=INTERRUPTION_COMMANDS,
        confidence_threshold=LOW_CONFIDENCE_THRESHOLD
    )
    
    # Setup Agent components (STT, LLM, TTS, VAD)
    agent = Agent(instructions="You are a friendly agent that pauses gracefully for real interruptions but ignores filler words while speaking.")
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"), # Deepgram is used here as it provides transcription events
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )

    await session.start(agent=agent, room=room)

    # 3. Custom Interruption Logic Hook
    # The handling should be done as an extension layer[cite: 12].
    @session.on("user_speech_transcribed")
    def handle_transcription(transcript_event):
        """Processes ASR transcription events from the STT plugin."""
        
        # Use getattr for safety, as field names can vary by plugin.
        transcript_text = transcript_event.text 
        transcript_confidence = getattr(transcript_event, 'confidence', 1.0) 
        
        # Check if TTS is currently active
        agent_is_speaking = session.tts_stream.is_active if session.tts_stream else False

        if interrupt_handler.should_interrupt(transcript_text, transcript_confidence, agent_is_speaking):
            
            # If the handler decides it's a valid interruption and the agent is speaking:
            if agent_is_speaking:
                # Gracefully pause when genuine interruptions are detected[cite: 58].
                print("Stopping agent streams due to valid interruption...")
                session.tts_stream.stop()
                session.llm_session.stop()
                
                # Queue the transcribed text as a new user turn to be processed by the LLM
                session.queue_user_turn(transcript_text)
            # If the agent is quiet, the event is registered and the core VAD/ASR loop continues the turn.
        
    # 4. Start the conversation flow
    await session.generate_reply(instructions="greet the user and tell them you are ready to talk. Ask them to try interrupting you with 'umm' and then with 'stop'.")
    
if __name__ == "__main__":
    # Ensure you set required environment variables (DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVEN_API_KEY)
    cli.run_app(entrypoint_fnc=entrypoint)