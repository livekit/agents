# examples/filler_aware_agent.py
# ---------------------------------------------------------------------------
# FINAL ENTRYPOINT FILE FOR SUBMISSION
# ---------------------------------------------------------------------------

import os
import asyncio
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import deepgram, silero, openai, cartesia

# ✅ Import middleware + dynamic keyword lists
from salescode_interrupts.interrupt_filter import (
    ASRSegment,
    IGNORED_FILLERS,
    INTERRUPT_COMMANDS
)
from salescode_interrupts.commands import build_interrupt_filter


# ---------------------------------------------------------------------------
# ✅ Conversational Agent
# ---------------------------------------------------------------------------
class FillerAwareAgent(Agent):
    def __init__(self):
        instructions = (
            f"You are a real-time conversational AI assistant. "
            f"While you are speaking, ignore filler sounds such as: "
            f"{', '.join(IGNORED_FILLERS)}. "
            f"Immediately stop speaking when the user says any command like: "
            f"{', '.join(INTERRUPT_COMMANDS)}. "
            f"Keep responses natural, concise, context-aware, and friendly. "
            f"Maintain a smooth conversation without reacting to background noise "
            f"or accidental hesitations."
        )

        super().__init__(instructions=instructions)

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and ask how their day is going."
        )


# ---------------------------------------------------------------------------
# ✅ LiveKit Entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # ✅ Load API keys
    DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    CARTESIA_KEY = os.getenv("CARTESIA_API_KEY")

    if not DEEPGRAM_KEY:
        raise ValueError("Missing DEEPGRAM_API_KEY")
    if not OPENAI_KEY:
        raise ValueError("Missing OPENAI_API_KEY")
    if not CARTESIA_KEY:
        raise ValueError("Missing CARTESIA_API_KEY")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2", api_key=DEEPGRAM_KEY),
        llm=openai.LLM(model="gpt-4o-mini", api_key=OPENAI_KEY),
        tts=cartesia.TTS(voice="maya", api_key=CARTESIA_KEY),
    )

    session.is_tts_active = False

    # ✅ Install the interruption filter 
    interrupt_filter = build_interrupt_filter(session)

    # -----------------------------------------------------------------------
    # ✅ ASR Events
    # -----------------------------------------------------------------------
    async def _async_asr_handler(evt):
        seg = ASRSegment(
            text=evt.text,
            is_final=evt.is_final,
            confidence=evt.confidence,
            language=getattr(evt, "language", "en"),
        )
        await interrupt_filter.handle_asr_segment(seg)

    def on_asr(evt):
        asyncio.create_task(_async_asr_handler(evt))

    session.on("stt.partial", on_asr)
    session.on("stt.final", on_asr)

    # -----------------------------------------------------------------------
    # ✅ TTS Events
    # -----------------------------------------------------------------------
    def on_tts_start(_):
        session.is_tts_active = True

    def on_tts_end(_):
        session.is_tts_active = False

    session.on("tts.start", on_tts_start)
    session.on("tts.end", on_tts_end)

    # -----------------------------------------------------------------------
    # ✅ Start Agent
    # -----------------------------------------------------------------------
    await session.start(agent=FillerAwareAgent(), room=ctx.room)


# ---------------------------------------------------------------------------
# ✅ Run from CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
