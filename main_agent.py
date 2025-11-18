# main_agent.py - main entrypoint for your voice agent
from __future__ import annotations

import logging
import asyncio  # NEW: needed for asyncio.create_task

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    UserInputTranscribedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ✅ Updated imports to use chat_processor
from chat_processor.agent_config import get_settings
from chat_processor.memory_manager import AgentStateTracker
from chat_processor.intent_detector import classify_transcript
from chat_processor.settings_loader import RuntimeWordConfig, watch_config_file

# Load .env so LIVEKIT_* and OPENAI_API_KEY are available
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. "
                "You speak clearly and keep your answers short."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    """
    This function is called by LiveKit to start your agent.
    It sets up the AgentSession and our filler-aware interruption logic.
    """
    settings = get_settings()
    logger.info("Starting session")
    logger.info("Ignored filler words (global): %s", settings.ignored_filler_words)
    logger.info(
        "Interrupt command words (global): %s", settings.interrupt_command_words
    )
    logger.info("Default language: %s", settings.default_language)

    # Runtime config object that knows per-language word lists.
    runtime_config = RuntimeWordConfig(
        ignored_by_lang=settings.ignored_filler_words_by_lang,
        commands_by_lang=settings.interrupt_command_words_by_lang,
        default_language=settings.default_language,
    )

    # OPTIONAL BONUS: watch a JSON file for dynamic updates if configured
    if settings.dynamic_config_path:
        logger.info(
            "Dynamic filler config enabled. Watching: %s",
            settings.dynamic_config_path,
        )
        asyncio.create_task(
            watch_config_file(settings.dynamic_config_path, runtime_config)
        )
    else:
        logger.info("Dynamic filler config not enabled (FILLER_CONFIG_PATH not set).")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_interruption_words=2,
    )

    state_tracker = AgentStateTracker()

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        state_tracker.update_agent_state(event.new_state)
        logger.info("Agent state: %s -> %s", event.old_state, event.new_state)

    @session.on("user_state_changed")
    def on_user_state_changed(event):
        state_tracker.update_user_state(event.new_state)
        logger.info("User state: %s -> %s", event.old_state, event.new_state)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        agent_speaking = state_tracker.is_agent_speaking()
        lang = (event.language or settings.default_language).split("-")[0].lower()
        ignored_words, command_words = runtime_config.get_sets_for_language(lang)

        decision = classify_transcript(
            transcript=event.transcript,
            agent_speaking=agent_speaking,
            is_final=event.is_final,
            ignored_filler_words=ignored_words,
            interrupt_command_words=command_words,
            confidence=None,
        )

        logger.info(
            "Transcript='%s' (final=%s, agent_speaking=%s, lang=%s) => %s",
            event.transcript,
            event.is_final,
            agent_speaking,
            lang,
            decision,
        )

        if decision == "interrupt_agent" and agent_speaking:
            logger.info(">> Interrupting agent due to real user interruption")
            session.interrupt()
        elif decision == "ignore_filler":
            logger.info(">> Ignoring filler while agent is speaking")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user briefly and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
