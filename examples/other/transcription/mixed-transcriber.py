import logging

from dotenv import load_dotenv

from livekit.agents import (
    AgentServer,
    AutoSubscribe,
    JobContext,
    TranscriptionSession,
    TranscriptSegment,
    cli,
    inference,
)

load_dotenv()

logger = logging.getLogger("mixed-transcriber")


# This example transcribes every participant in the room through a single STT stream.
# Participants' audio is mixed, and each word is attributed to whoever was speaking when it was
# said, so the room pays for one STT connection rather than one per participant.
# Compare multi-user-transcriber.py, which runs a separate session per participant.

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = TranscriptionSession(stt=inference.STT("deepgram/nova-3"))

    @session.on("transcript_segment")
    def _on_segment(segment: TranscriptSegment) -> None:
        if segment.final and segment.text:
            logger.info(f"{segment.participant_identity} -> {segment.text}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await session.start(room=ctx.room)
    ctx.add_shutdown_callback(session.aclose)


if __name__ == "__main__":
    cli.run_app(server)
