import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.agents.utils import AgentState # <-- Required for state tracking

from .intelligent_handler import IntelligentInterruptionHandler 


logger = logging.getLogger("resume-agent")
load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # 1. Define configuration and initialize handler
    FILLER_WORDS = ['uh', 'umm', 'hmm', 'haan']
    handler = IntelligentInterruptionHandler(FILLER_WORDS)
    
    # 2. Existing AgentSession setup:
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        # Keep these settings—we are enhancing, not replacing, the timeout logic
        false_interruption_timeout=1.0,
        resume_false_interruption=True,
    )
    
    # 3. HOOK 1: Track Agent State (TTS Start/Stop)
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        is_speaking = (event.new_state == AgentState.SPEAKING)
        handler.update_agent_speaking_status(is_speaking)
        logger.info(f"Agent State: {event.new_state.name}. Speaking status: {is_speaking}")


    # 4. HOOK 2: Filter User Transcription (Core Interruption Logic)
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        text = event.transcript
        confidence = event.confidence
        
        should_interrupt = handler.should_interrupt(text, confidence)

        if should_interrupt:
            # Case: Genuine interruption or speech while quiet. Force stop.
            session.interrupt() 
            logger.warning(f"ACTION: Stopping agent due to VALID interruption: '{text}'")
        else:
            # Case: Filler while speaking or low confidence. IGNORE the event.
            logger.info(f"IGNORED: Classified as filler/low confidence: '{text}'")
            # Do nothing else, allowing the false_interruption_timeout to handle the resume.
            pass
            
    # 5. START THE SESSION (CRITICAL MISSING LINE)
    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)