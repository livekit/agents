import logging
import os
from livekit.agents import AgentSession, cli, JobContext, WorkerOptions

# ✅ Plugins
from livekit.plugins import deepgram, cartesia, silero, openai

from custom_agents.interrupt_handler_agent import InterruptHandlerAgent

logging.basicConfig(level=logging.INFO)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = InterruptHandlerAgent(
        instructions=(
            "You are a helpful voice assistant. Ignore filler-only interruptions "
            "while speaking; stop immediately on commands like wait/stop."
        )
    )

    # ✅ Build plugin-based components
    vad = silero.VAD.load()
    stt = deepgram.STTv2(model="flux-general-en", eager_eot_threshold=0.4,api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = openai.LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    tts = cartesia.TTS(model="sonic-3", voice=os.getenv("CARTESIA_API_KEY"))

    session = AgentSession(
        turn_detection="vad",
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        allow_interruptions=True,
        resume_false_interruption=True,
        false_interruption_timeout=0.2,
    )

    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
