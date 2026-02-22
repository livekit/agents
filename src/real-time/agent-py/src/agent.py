import logging
import os
import signal
import sys
from typing import Optional

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    metrics,
    room_io,
)
from livekit.plugins import (
    anam,
    noise_cancellation,
    openai,
)

logger = logging.getLogger("real-time-agent")
logger.setLevel(logging.DEBUG)

load_dotenv(".env.local")

# Configuration: Choose between "realtime" (OpenAI Realtime API) or "inference" (LiveKit Inference)
USE_OPENAI_REALTIME = False

# Handle graceful shutdown on Ctrl+C
def signal_handler(sig, frame):
    print("\n\nAgent stopped gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class RealtimeAgent(Agent):
    """Real-time agent with vision and avatar support."""
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful AI assistant that can see the user's video and hear their voice. "
            "You provide thoughtful responses and engage naturally with the user. "
            "When you see the user, acknowledge what you observe and respond helpfully."
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    """Preload components to reduce startup time."""
    if not USE_OPENAI_REALTIME:
        from livekit.plugins import silero
        proc.userdata["vad"] = silero.VAD.load()
        logger.debug("VAD preloaded")


server.setup_fnc = prewarm


@server.rtc_session(agent_name="real-time-agent")
async def entrypoint(ctx: JobContext):
    """Main agent entrypoint for real-time vision and avatar interactions."""
    logger.info(f"Starting real-time agent session for room: {ctx.room.name}")
    
    # Validate required environment variables
    anam_api_key = os.getenv("ANAM_API_KEY")
    if not anam_api_key:
        logger.error("ANAM_API_KEY is not set")
        raise ValueError("ANAM_API_KEY is not set")
    
    anam_avatar_id = os.getenv("ANAM_AVATAR_ID")
    if not anam_avatar_id:
        logger.error("ANAM_AVATAR_ID is not set")
        raise ValueError("ANAM_AVATAR_ID is not set")
    
    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    logger.debug("Creating agent session...")
    
    if USE_OPENAI_REALTIME:
        logger.debug("Using OpenAI Realtime API")
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(voice="coral"),
        )
    else:
        from livekit.agents import inference
        from livekit.plugins.turn_detector.multilingual import MultilingualModel
        
        logger.debug("Using LiveKit Inference pipeline with GPT-4o-vision")
        session = AgentSession(
            stt=inference.STT(model="deepgram/nova-3", language="multi"),
            llm=inference.LLM(model="openai/gpt-4o-mini"),
            tts=inference.TTS(
                model="elevenlabs/eleven_turbo_v2_5",
                voice="cgSgspJ2msm6clMCkdW9",  # Jessica (ElevenLabs)
                sample_rate=16000,  # CRITICAL: Must be 16kHz for Anam avatar lip-sync
            ),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )
    
    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)
    
    logger.debug("Starting session with voice and video pipeline...")
    await session.start(
        agent=RealtimeAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
            # Enable video input for vision capabilities
            video_input=True,
        ),
    )
    
    logger.debug(f"Initializing Anam avatar with ID: {anam_avatar_id}")
    avatar = anam.AvatarSession(
        persona_config=anam.PersonaConfig(
            name="AI Assistant",
            avatarId=anam_avatar_id,
        ),
        api_key=anam_api_key,
    )

    logger.info("Starting Anam avatar session...")
    await avatar.start(session, room=ctx.room)
    logger.info("Avatar session started successfully")

    logger.debug("Generating initial greeting...")
    session.generate_reply(
        instructions="Greet the user warmly and offer your assistance. Acknowledge if you can see them."
    )
    
    logger.info("Agent session initialized and running")


if __name__ == "__main__":
    cli.run_app(server)
