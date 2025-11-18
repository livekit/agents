import logging
from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import cartesia, deepgram, openai, silero

from intelligent_handler import IntelligentInterruptionHandler  # adjust if needed



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("resume-agent")

# Load env variables
load_dotenv()

server = AgentServer()


# ----------------------  VERSION-SAFE SPEAKING CHECK ----------------------
def _is_speaking_state(state_obj) -> bool:
    """
    Detects if the agent is in SPEAKING state without relying on AgentState import.
    Works across all LiveKit versions (enum, string, or custom state objects).
    """
    if state_obj is None:
        return False

    # Case 1: Enum-like object with .name
    name = getattr(state_obj, "name", None)
    if isinstance(name, str) and name.upper() == "SPEAKING":
        return True

    # Case 2: Direct string representation
    try:
        sval = str(state_obj).lower()
        if "speaking" in sval:
            return True
    except:
        pass

    # Case 3: Enum with .value attribute
    value = getattr(state_obj, "value", None)
    if isinstance(value, str) and value.lower() == "speaking":
        return True

    return False


# ----------------------  MAIN SESSION ----------------------
@server.rtc_session()
async def entrypoint(ctx: JobContext):

    logger.info(" Interruption-safe voice agent initializing...")

    # Filler words
    FILLER_WORDS = ["uh", "umm", "hmm", "haan"]
    handler = IntelligentInterruptionHandler(FILLER_WORDS)

    # Configure the session
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(voice="maya"),

        # Provided logic from your instructions
        false_interruption_timeout=1.0,
        resume_false_interruption=True
    )

    # -------------------------------- HOOK 1 --------------------------------
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        is_speaking = _is_speaking_state(event.new_state)
        handler.update_agent_speaking_status(is_speaking)

        logger.info(
            f" Agent State Change → {getattr(event.new_state, 'name', str(event.new_state))} | speaking={is_speaking}"
        )

    # -------------------------------- HOOK 2 --------------------------------
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):

        # Ignore partial transcripts
        if not event.is_final:
            return

        text = event.transcript.strip()
        conf = event.confidence

        logger.info(f" User said: '{text}' | confidence={conf:.2f}")

        should_interrupt = handler.should_interrupt(text, conf)

        if should_interrupt:
            logger.warning(f" VALID INTERRUPTION: '{text}' → stopping agent")
            session.interrupt()
        else:
            logger.info(f" IGNORED as filler/noise: '{text}'")

    # ----------------------------- START SESSION -----------------------------
    logger.info(" Starting agent session...")

    await session.start(
        agent=Agent(
            instructions="You are a natural and helpful assistant. Do not stop speaking for fillers."
        ),
        room=ctx.room
    )

    logger.info(" Agent session started successfully.")


if __name__ == "__main__":
    cli.run_app(server)
