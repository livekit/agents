import logging
from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, groq, silero

load_dotenv()

logger = logging.getLogger("realtime-with-tts")
logger.setLevel(logging.INFO)

class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful and concise assistant.",
            # 1. Hearing (Deepgram STT)
            stt=deepgram.STT(),
            
            # 2. Thinking (Groq LLM - Llama 3)
            llm=groq.LLM(model="llama-3.1-8b-instant"),
            
            # 3. Speaking (Deepgram TTS)
            tts=deepgram.TTS(),
            
            # 4. Interruption Detection (Silero VAD)
            vad=silero.VAD.load()
        )

    @function_tool
    async def get_weather(self, location: str):
        """Called when the user asks about the weather."""
        logger.info(f"getting weather for {location}")
        return f"The weather in {location} is sunny, and the temperature is 20 degrees Celsius."

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Connect to the room
    await ctx.connect()
    
    # Create the session
    session = AgentSession()

    # Start the agent with the room
    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=True,
            audio_output=True,
        ),
    )
    
    # Say hello
    session.generate_reply(instructions="say hello to the user")

if __name__ == "__main__":
    cli.run_app(server)
