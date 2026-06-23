import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    cli,
    inference,
    room_io,
)
from livekit.plugins import rumik_ai

try:
    # Silero VAD gives fast end-of-turn detection (like the Rumik Pipecat example).

    # Without it the session falls back to slower STT endpointing, which is the main
    # source of response latency. Optional so the example still runs if it's not
    # installed: `uv pip install -e livekit-plugins/livekit-plugins-silero onnxruntime`.
    from livekit.plugins import silero
except ImportError:
    silero = None

logger = logging.getLogger("rumik-ai-agent")
logger.setLevel(logging.INFO)

load_dotenv()

FRIEND_NAME = os.getenv("RUMIK_AI_FRIEND_NAME", "Mira")

BASE_COMPANION_INSTRUCTIONS = f"""
You are {FRIEND_NAME}, a warm friend on a voice call.
You are here to listen, understand, and ask gentle questions about the user's life.

Conversation style:
- Sound like a real friend, not a coach, therapist, or formal assistant.
- Listen first. Reflect what the user said before asking the next question.
- Ask one gentle question at a time.
- Take care of the pronouns for yourself in hinglish. Use "main" for "I/me" and "tum" for "you". like "main samajh sakti hoon" not "main samajh sakta hoon".
- Ask about normal life topics: day, work, family, friends, stress, dreams, plans, memories.
- Do not rush to fix the user's problem. Avoid long advice unless they ask for it.
- If the user seems sad, stressed, or confused, acknowledge it softly and stay with them.
- If the user mentions immediate danger or self-harm, encourage them to contact a trusted person
  or local emergency help right away.
- Keep every reply short enough for natural speech, usually one or two sentences.
- Mix simple Hindi with common English words like day, work, stress, family, friends, plan,
  mood, break, and feeling.
- Do not reply in full English, even if the user speaks English.
- Bad style: "How was your day and what emotional concern should we analyze?"
- Do not use markdown, bullets, or emojis.
""".strip()

# Script style is model-specific: muga is trained on Romanized Hinglish (Roman script
# only), while mulberry expects Hindi in Devanagari with English words kept in Latin script.
MUGA_SCRIPT_INSTRUCTIONS = """
Script:
- Speak in natural Romanized Hinglish only (Roman/Latin script).
- Do not use Devanagari or any other Indian script.
- Good style: "Aaj ka din kaisa gaya? Kuch heavy lag raha hai kya?"
""".strip()

MULBERRY_SCRIPT_INSTRUCTIONS = """
Script:
- Write Hindi words in Devanagari, and keep English words in Latin script as-is (do not
  transliterate English into Devanagari).
- Good style: "आज का din कैसा रहा? कुछ heavy लग रहा है क्या?"
""".strip()

MUGA_PROMPTING_INSTRUCTIONS = """
Muga TTS prompting instructions:
- Every spoken reply must start with exactly one lowercase global tone tag.
- Supported tags are [happy], [excited], [sad], [angry], [neutral], and [whisper].
- Choose the tag that best matches the whole reply.
- Put exactly one space after the closing bracket.
- Use [neutral] for calm listening, simple questions, and grounded reflection.
- Use [happy] for warm greetings, friendly check-ins, and light moments.
- Use [excited] only for genuine high-energy moments, such as good news or a happy plan.
- Use [sad] for soft empathy when the user shares something difficult.
- Use [angry] only if the user is frustrated and you need a firm but controlled response.
- Use [whisper] only for quiet, private, or careful phrasing.
- Supported event tags are <laugh>, <chuckle>, and <sigh>.
- Use events sparingly and only when natural.
- Use <laugh> only with [happy] or [excited].
- Use <chuckle> only with [happy], [excited], or [whisper].
- Use <sigh> only with [sad], [angry], [neutral], or [whisper].
""".strip()


def _agent_instructions(tts_model: str) -> str:
    if tts_model == "muga":
        return (
            f"{BASE_COMPANION_INSTRUCTIONS}\n\n{MUGA_SCRIPT_INSTRUCTIONS}\n\n"
            f"{MUGA_PROMPTING_INSTRUCTIONS}"
        )
    return f"{BASE_COMPANION_INSTRUCTIONS}\n\n{MULBERRY_SCRIPT_INSTRUCTIONS}"


class RumikAICompanionAgent(Agent):
    def __init__(self, *, tts_model: str) -> None:
        self._tts_model = tts_model
        super().__init__(instructions=_agent_instructions(tts_model))

    async def on_enter(self) -> None:
        if self._tts_model == "mulberry":
            instructions = (
                f"Warmly greet the user as {FRIEND_NAME} in Hindi (Devanagari) with English "
                "words kept in Latin script, then ask how their day was."
            )
        else:
            instructions = (
                f"Romanized Hinglish mein user ko warmly greet karo as {FRIEND_NAME}, "
                "phir pucho ki unka din kaisa gaya. "
                "Start with [happy]."
            )
        self.session.generate_reply(instructions=instructions)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    # Load the VAD once per worker process so turn detection is ready before the first
    # turn (no model-load latency on the first response).
    if silero is not None:
        proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def _rumik_tts() -> rumik_ai.TTS:
    model = os.getenv("RUMIK_AI_MODEL", "muga")

    # Point at a custom Rumik gateway (e.g. staging) via RUMIK_GATEWAY_URL; otherwise
    # the plugin's default base URL is used.
    gateway_url = os.getenv("RUMIK_GATEWAY_URL")
    common: dict[str, str] = {"base_url": gateway_url} if gateway_url else {}

    # muga buffers the full reply by default so its [tone] tag conditions the whole
    # utterance; mulberry streams sentence-by-sentence by default for lower latency.
    # Set RUMIK_AI_FULL_RESPONSE=0/1 to override that per-model default.
    agg: dict[str, bool] = {}
    raw_full_response = os.getenv("RUMIK_AI_FULL_RESPONSE")
    if raw_full_response is not None:
        agg["full_response_aggregation"] = raw_full_response.lower() not in {"0", "false", "no"}

    if model == "mulberry":
        # Mulberry is steered with a voice description or preset speaker, not tone tags.
        # Only pass speaker when set -- the plugin rejects None as an invalid speaker.
        speaker = os.getenv("RUMIK_AI_SPEAKER")
        if speaker:
            common["speaker"] = speaker
        return rumik_ai.TTS(
            model="mulberry",
            description=os.getenv(
                "RUMIK_AI_DESCRIPTION",
                "warm, gentle female friend",
            ),
            f0_up_key=float(os.getenv("RUMIK_AI_F0_UP_KEY", "0")),
            **agg,
            **common,
        )

    # Muga expects the LLM to include one global tone tag, unless RUMIK_AI_TONE is set.
    return rumik_ai.TTS(
        model="muga",
        tone=os.getenv("RUMIK_AI_TONE") or None,
        **agg,
        **common,
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    tts_model = os.getenv("RUMIK_AI_MODEL", "muga")
    stt_model = os.getenv("LIVEKIT_STT_MODEL", "deepgram/nova-3")
    stt_language = os.getenv("LIVEKIT_STT_LANGUAGE", "multi")
    llm_model = os.getenv("LIVEKIT_LLM_MODEL", "openai/gpt-4.1-mini")

    logger.info(
        "Starting STT -> LLM -> rumik-ai TTS pipeline: stt=%s language=%s llm=%s tts=%s",
        stt_model,
        stt_language,
        llm_model,
        tts_model,
    )

    # AgentSession owns the STT -> LLM -> TTS loop. The app only wires providers together.
    session = AgentSession(
        stt=inference.STT(
            model=stt_model,
            language=stt_language,
        ),
        llm=inference.LLM(model=llm_model),
        tts=_rumik_tts(),
        # Fast end-of-turn detection; None falls back to STT endpointing (slower).
        vad=ctx.proc.userdata.get("vad"),
    )

    @session.on("session_usage_updated")
    def _on_session_usage_updated(ev: SessionUsageUpdatedEvent) -> None:
        logger.info("Session usage updated: %s", ev.usage)

    async def log_usage() -> None:
        logger.info("Usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=RumikAICompanionAgent(tts_model=tts_model),
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_input=room_io.AudioInputOptions()),
        record=False,
    )


if __name__ == "__main__":
    cli.run_app(server)
