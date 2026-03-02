"""
Example demonstrating ElevenLabs eleven_v3 TTS model usage with LiveKit Agents.

The eleven_v3 model doesn't support WebSocket streaming, so the plugin automatically
uses HTTP streaming with chunked transfer encoding instead.

To compare latency between models:
- Set USE_V3 = True  → Uses eleven_v3 with HTTP streaming (new)
- Set USE_V3 = False → Uses eleven_turbo_v2_5 with WebSocket (existing)

Look for the "tts_ttfb" metric in the console to compare TTS latency.
"""

from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import elevenlabs, openai, silero

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Toggle between models to compare latency:
    # - eleven_v3: Uses HTTP streaming (our new implementation)
    # - eleven_turbo_v2_5: Uses WebSocket streaming (existing)

    USE_V3 = True  # Set to False to test eleven_turbo_v2_5

    agent = Agent(
        instructions="You are a helpful voice assistant. Keep responses very short - 1 sentence max.",
        stt=openai.STT(),  # OpenAI Whisper
        llm=openai.LLM(model="gpt-4o"),  # Faster than gpt-4o-mini for streaming
        tts=elevenlabs.TTS(
            model="eleven_v3" if USE_V3 else "eleven_turbo_v2_5",
            voice_id="EXAVITQu4vr4xnSDxMaL",
        ),
        vad=silero.VAD.load(),
    )

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
