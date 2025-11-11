import logging
import asyncio
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")

load_dotenv(".env.local")


# ---------------------- Agent Definition ----------------------
class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful, concise voice assistant. "
                         "Keep replies natural, short, and clear."
        )


# ---------------------- Prewarm ----------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarming embedding model...")
    proc.userdata["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded.")


# ---------------------- Entrypoint ----------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Create session
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        allow_interruptions=True,          # keep STT active
        min_interruption_duration=9999,    # prevent auto interruption
        min_interruption_words=9999,       # prevent STT auto interruption
    )

    # ---------------------- Metrics ----------------------
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
        room_output_options=RoomOutputOptions(sync_transcription=False),
    )

    # ---------------------- Embeddings Setup ----------------------
    embedder: SentenceTransformer = ctx.proc.userdata.get("embedder")

    INTERRUPT_PHRASES = [
        "stop", "cancel", "no", "wait", "hold on", "pause", "enough",
        "that's fine", "okay stop", "please stop", "no that's good",
        "not needed", "stop talking", "hold please", "please don't",
        "please stop talking",
    ]

    FILLER_PHRASES = [
        "", "yes", "go on", "nice",
        "hmm", "umm", "uh", "ah", "haan", "huh", "hmmm", "mmm", "mm",
        "uhh", "erm", "ermm", "ahh",
    ]

    INTERRUPT_EMBS = embedder.encode(INTERRUPT_PHRASES, normalize_embeddings=True)
    FILLER_EMBS = embedder.encode(FILLER_PHRASES, normalize_embeddings=True)

    INTERRUPT_SIM_THRESHOLD = 0.78
    FILLER_SIM_THRESHOLD = 0.65

    # ---------------------- Turn Detector Mute Trick ----------------------
    orig_turn_detection = session.turn_detection

    def mute_turn_detection():
        logger.debug("🔇 Turn detector muted during agent speech")
        session.turn_detection = None

    def restore_turn_detection():
        logger.debug("🔊 Turn detector restored")
        session.turn_detection = orig_turn_detection

    session.on("agent_speaking_started", lambda _: mute_turn_detection())
    session.on("agent_speaking_stopped", lambda _: restore_turn_detection())

    # ---------------------- Transcription Handler ----------------------
    async def _on_user_input_async(ev):
        text = getattr(ev, "transcript", None)
        if not text:
            return
        text = text.strip().lower()

        # Ignore blanks
        if not text:
            logger.info("🟡 Filler (empty transcript)")
            return

        logger.debug(f"👂 User said: '{text}'")

        # Compute embeddings
        text_emb = embedder.encode([text], normalize_embeddings=True)[0]

        # Filler detection
        filler_sim = float(np.max(np.dot(FILLER_EMBS, text_emb)))
        if filler_sim >= FILLER_SIM_THRESHOLD:
            logger.info(f"🟡 FILLER DETECTED (sim={filler_sim:.3f}) → '{text}' ignored")
            return

        # Interrupt detection
        interrupt_sim = float(np.max(np.dot(INTERRUPT_EMBS, text_emb)))
        if interrupt_sim >= INTERRUPT_SIM_THRESHOLD:
            logger.info(f"🔴 STOP DETECTED (sim={interrupt_sim:.3f}) → '{text}'")
            await session.interrupt()
        else:
            logger.info(f"⚪ No match → filler_sim={filler_sim:.3f}, interrupt_sim={interrupt_sim:.3f}")

    def _on_user_input_sync(ev):
        try:
            asyncio.create_task(_on_user_input_async(ev))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.create_task(_on_user_input_async(ev))

    # Subscribe to user text events
    try:
        session.on("user_input_transcribed", _on_user_input_sync)
        logger.info("✅ Listening for user_input_transcribed event")
    except Exception as e:
        logger.warning(f"Failed to register event listener: {e}")

    # ---------------------- Example Output ----------------------
    handle = await session.say(
        "Hi! I’m active. Speak while I talk — I’ll ignore hmm or umm but stop if you say cancel or stop."
    )

    await ctx.connect()


# ---------------------- Run Worker ----------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=120.0,
        )
    )
