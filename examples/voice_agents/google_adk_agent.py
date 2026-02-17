"""Example: Using a Google ADK agent as the LLM in a LiveKit voice agent.

ADK handles tool calling and multi-agent orchestration internally, while
LiveKit manages voice orchestration (turns, interruptions, STT/TTS).

Dependencies:
    pip install livekit-agents[silero,deepgram,turn_detector] \
                livekit-plugins-google-adk google-adk
"""

import logging

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, inference
from livekit.plugins import google_adk, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("adk-agent")

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# --- Define ADK agent with tools ---


def get_weather(city: str) -> dict:
    """Returns weather information for the given city."""
    return {"city": city, "temp": "72°F", "condition": "Sunny"}


adk_agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful voice assistant. Keep responses concise "
        "since your output will be spoken aloud via TTS."
    ),
    tools=[get_weather],
)


# --- LiveKit entrypoint ---


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    agent = Agent(
        instructions="",
        llm=google_adk.LLMAdapter(adk_agent),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=inference.STT("deepgram/nova-3", language="multi"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Greet the user warmly.")


if __name__ == "__main__":
    cli.run_app(server)
