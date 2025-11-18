from dotenv import load_dotenv
load_dotenv()

import json
import logging
import asyncio

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    UserInputTranscribedEvent,
    SpeechCreatedEvent,
)

# FREE MODELS
from livekit.plugins import deepgram, groq, cartesia, silero

# Your custom logic
from custom.interrupt_handler import InterruptHandler
from custom.filler_manager import FillerManager

# -----------------------------------------------------------
# LOGGING
# -----------------------------------------------------------
LOG_PATH = "logs/interrupt.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -----------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------
async def entrypoint(ctx: agents.JobContext) -> None:

    # Connect worker to the LiveKit room
    await ctx.connect()

    # -------------------------------------------------------
    # VOICE PIPELINE (with FREE providers)
    # -------------------------------------------------------
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        allow_interruptions=True,   # ✅ REQUIRED FOR USER INTERRUPTIONS
    )

    # Agent prompt settings
    agent = Agent(
        instructions=(
            "You are a concise, helpful voice assistant for a technical interview project. "
            "Keep your responses short, natural, and clear."
        ),
        allow_interruptions=True,   # ✅ REQUIRED
    )

    # -------------------------------------------------------
    # LOAD FILLERS + COMMAND WORDS
    # -------------------------------------------------------
    filler_manager = FillerManager(
        filler_file_path="config/fillers_default.json",
        reload_interval=30,
    )

    with open("config/commands_default.json", "r") as f:
        command_list = [c.lower().strip() for c in json.load(f)]

    interrupt_handler = InterruptHandler(
        filler_manager=filler_manager,
        command_list=command_list,
        confidence_threshold=0.6,
    )

    agent_speaking = False

    # -------------------------------------------------------
    # EVENT — AGENT SPEAKING (TTS)
    # -------------------------------------------------------
    @session.on("speech_created")
    def _on_speech_created(event: SpeechCreatedEvent):
        asyncio.create_task(handle_agent_speech(event))

    async def handle_agent_speech(event: SpeechCreatedEvent):
        nonlocal agent_speaking

        if getattr(event, "is_final", False):
            agent_speaking = False
            logging.info("[AGENT FINISHED SPEAKING]")
        else:
            if not agent_speaking:
                logging.info("[AGENT STARTED SPEAKING]")
            agent_speaking = True

    # -------------------------------------------------------
    # EVENT — USER SPEECH (STT)
    # -------------------------------------------------------
    @session.on("user_input_transcribed")
    def _on_user_input(event: UserInputTranscribedEvent):
        asyncio.create_task(handle_user_input(event))

    async def handle_user_input(event: UserInputTranscribedEvent):
        nonlocal agent_speaking

        # Only react on FINAL transcripts
        if not getattr(event, "is_final", True):
            return

        text = (event.transcript or "").lower().strip()
        if not text:
            return

        confidence = getattr(event, "confidence", 1.0)

        logging.info(
            f"[STT] {text} (conf={confidence}, agent_speaking={agent_speaking})"
        )

        # Decision via your logic
        decision = interrupt_handler.process_transcript(
            text=text,
            confidence=confidence,
            agent_speaking=agent_speaking,
        )

        # ----- IGNORE -----
        if decision == "ignore":
            logging.info(f"[IGNORED] {text}")
            return

        # ----- INTERRUPT -----
        if decision == "interrupt":
            logging.info("[INTERRUPT] User interrupted — stopping agent speech.")
            session.interrupt()  # NOW fully works because interrupts are enabled
            await session.generate_reply(user_input=text)
            return

        # ----- NORMAL -----
        if decision == "normal":
            logging.info(f"[NORMAL] replying → {text}")
            await session.generate_reply(user_input=text)
            return

    # -------------------------------------------------------
    # START AGENT SESSION
    # -------------------------------------------------------
    await session.start(agent=agent, room=ctx.room)

    # Optional greeting
    await session.generate_reply(
        instructions="Greet the user briefly and say you are ready."
    )
