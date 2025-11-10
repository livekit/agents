import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    MetricsCollectedEvent,
    RoomOutputOptions,
    StopResponse,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("transcriber")


# This example demonstrates how to transcribe audio from a remote participant.
# It creates agent session with only STT enabled and publishes transcripts to the room.


class Transcriber(Agent):
    def __init__(self):
        super().__init__(
            instructions="not-needed",
            stt=openai.STT(),
        )

    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content
        logger.info(f" -> {user_transcript}")

        raise StopResponse()


async def entrypoint(ctx: JobContext):
    logger.info(f"starting transcriber (speech to text) example, room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        # vad is needed for non-streaming STT implementations
        vad=silero.VAD.load(min_silence_duration=0.3),
    )

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent=Transcriber(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            # disable audio output if it's not needed
            audio_enabled=False,
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
