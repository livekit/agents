import os
import logging
import asyncio
import dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, cartesia

# Load environment variables
dotenv.load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleInterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a friendly and professional interview assistant. "
                        "Your role is to conduct mock interviews for software engineering positions. "
                        "Ask relevant technical and behavioral questions, listen carefully to responses, "
                        "and provide appropriate follow-up questions. "
                        "Be conversational, professional, and helpful. "
                        "Start by greeting the candidate and asking about their background."
        )

async def entrypoint(ctx: JobContext):
    """Simple interview agent without any user input requirements"""
    try:
        logger.info("Starting simple interview agent...")
        
        # Initialize components
        stt = cartesia.STT(
            model="sonic-2",
            language="en-US",
            enable_automatic_punctuation=True
        )

        tts = cartesia.TTS(
            model="sonic-2",
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
            language="en",
            speed=1.0
        )

        llm = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7
        )

        vad = cartesia.VAD()

        # Create agent session
        session = AgentSession(
            stt=stt,
            tts=tts,
            llm=llm,
            vad=vad,
            allow_interruptions=True,
            min_endpointing_delay=0.5,
            max_endpointing_delay=3.0
        )

        # Connect to the room
        await ctx.connect()
        logger.info(f"Connected to room: {ctx.room.name}")

        # Start the session
        agent = SimpleInterviewAgent()
        await session.start(room=ctx.room, agent=agent)

        # Generate initial greeting
        await session.generate_reply(
            instructions="Greet the candidate warmly. Introduce yourself as an AI interview assistant "
                        "and explain that you'll be conducting a mock interview for a software engineering position. "
                        "Start with a simple introductory question like 'Can you tell me about yourself?' "
                        "or 'What interests you about software development?'"
        )

        logger.info("Interview agent started successfully. Waiting for candidate...")

        # Keep the session active
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in interview session: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the agent
    opts = WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(opts)