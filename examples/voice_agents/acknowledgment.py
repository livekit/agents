import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    llm,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("acknowledgment-agent")

load_dotenv()


class AcknowledgmentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful and friendly assistant. Keep your responses concise."
        )

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """
        This callback is triggered when the user finishes speaking.
        We use it to demonstrate the blocking acknowledgment feature.
        """

        # 1. Start waiting for acknowledgment.
        # This blocks the speech queue with a high-priority handle.
        # It ensures that even if the LLM responds, its speech won't start
        # until the acknowledgment is either played or skipped.
        # The timeout prevents the queue from being blocked indefinitely.
        self.wait_for_acknowledgment(timeout=3.0)

        # 2. Simulate some backend work (e.g., database lookup, API call).
        # While this delay is happening, the AgentActivity is already starting the LLM generation in parallel.
        await asyncio.sleep(0.8)

        # 3. Set the acknowledgment message.
        # This signals the acknowledgment task to synthesize and play this message immediately.
        # If the LLM response had already arrived (very fast TTFT), this call would effectively
        # do nothing because the acknowledgment would have been automatically skipped.
        self.set_acknowledgment("Just a second, I'm checking that for you...")

        # The full LLM response will follow automatically once the acknowledgment message
        # finishes playing.


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-2",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    @session.on("speech_created")
    def _on_speech_created(ev):
        if ev.source == "acknowledgment":
            logger.info("Acknowledgment speech handle created")

            def _on_done(handle):
                if handle.interrupted:
                    logger.info("Acknowledgment was SKIPPED or TIMED OUT (LLM responded early)")
                else:
                    logger.info("Acknowledgment message PLAYED successfully")

            ev.speech_handle.add_done_callback(_on_done)

    await session.start(
        agent=AcknowledgmentAgent(),
        room=ctx.room,
    )
    logger.info("Agent started")


if __name__ == "__main__":
    cli.run_app(server)
