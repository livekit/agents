import logging

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.plugins import google_adk, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("google-adk-agent")

load_dotenv()

# This example demonstrates adding voice to a Google ADK agent by using our
# adapter. Instructions and tool calls are handled by ADK, while voice
# orchestration (turns, interruptions, etc) is handled by the Agents framework.
#
# In order to run this example, you need:
#   pip install google-adk livekit-agents[silero,google_adk,deepgram,turn_detector]
#
# Set GOOGLE_API_KEY in your environment (or .env file).

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name to look up weather for.

    Returns:
        A dictionary with weather information.
    """
    return {"city": city, "temperature": "72°F", "condition": "sunny"}


def create_runner() -> Runner:
    adk_agent = LlmAgent(
        name="weather_assistant",
        model="gemini-2.5-flash",
        instruction="You are a helpful voice assistant. Be concise in your responses.",
        tools=[get_weather],
    )
    return Runner(
        agent=adk_agent,
        app_name="livekit-adk-example",
        session_service=InMemorySessionService(),
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    runner = create_runner()

    agent = Agent(
        # instructions are handled by the ADK agent, leave empty here
        instructions="",
        llm=google_adk.LLMAdapter(runner),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=inference.STT("deepgram/nova-3", language="multi"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
    )
    await session.generate_reply(instructions="Greet the user and ask how you can help them today.")


if __name__ == "__main__":
    cli.run_app(server)
