"""Local console harness for testing Fish Audio expressive mode.

Run with local mic/speaker (no LiveKit server needed):

    uv run python examples/expressive_fish_console.py console

Everything you'd want to A/B while picking voices is env-driven so you can
swap without editing code:

    FISH_VOICE_ID   Fish reference_id (voice model). Defaults to the plugin default.
    PRESET          conversational | customer_service | healthcare  (default: conversational)
    GROQ_LLM        Groq LLM model (default: openai/gpt-oss-120b)
    FISH_MODEL      s2-pro | s1  (default: s2-pro)
    FISH_SPEED      speaking rate (1.0 normal, <1 slower, >1 faster). Unset = natural.
    FISH_VOLUME     loudness in dB (0 = natural). Unset = unchanged.

Required keys in a .env file (or the environment):

    GROQ_API_KEY=...
    FISH_API_KEY=...
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice import presets
from livekit.plugins import fishaudio, groq, silero

logger = logging.getLogger("expressive-fish")
logger.setLevel(logging.INFO)

load_dotenv()

_PRESETS = {
    "conversational": presets.CONVERSATIONAL,
    "customer_service": presets.CUSTOMER_SERVICE,
    "healthcare": presets.HEALTHCARE,
}

PRESET = os.getenv("PRESET", "conversational").lower()
GROQ_LLM = os.getenv("GROQ_LLM", "openai/gpt-oss-120b")
FISH_MODEL = os.getenv("FISH_MODEL", "s2-pro")
FISH_VOICE_ID = os.getenv("FISH_VOICE_ID")  # None -> plugin default voice
FISH_SPEED = os.getenv("FISH_SPEED")  # None -> voice's natural pace
FISH_VOLUME = os.getenv("FISH_VOLUME")  # None -> unchanged

INSTRUCTIONS = (
    "You're a friendly, expressive voice agent being used to demo what the "
    "Fish Audio voice can do. Keep replies short and natural. Lean into the "
    "emotional range — react genuinely, vary your energy, and let feeling come "
    "through. If the conversation lulls, offer to tell a short joke or a quick "
    "story so the listener can hear a full range of delivery."
)


class ExpressiveAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the user warmly and with real excitement, and ask how "
                "their day is going. One or two short sentences."
            )
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    expressive = _PRESETS.get(PRESET, presets.CONVERSATIONAL)
    logger.info(
        "starting: preset=%s llm=%s fish_model=%s voice=%s",
        PRESET,
        GROQ_LLM,
        FISH_MODEL,
        FISH_VOICE_ID or "(default)",
    )

    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=groq.LLM(model=GROQ_LLM),
        tts=fishaudio.TTS(
            model=FISH_MODEL,
            voice_id=FISH_VOICE_ID if FISH_VOICE_ID else NOT_GIVEN,
            speed=float(FISH_SPEED) if FISH_SPEED else NOT_GIVEN,
            volume=float(FISH_VOLUME) if FISH_VOLUME else NOT_GIVEN,
        ),
        vad=silero.VAD.load(),
        expressive=expressive,
    )

    await session.start(agent=ExpressiveAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
