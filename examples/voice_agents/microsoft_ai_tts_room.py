"""Say one sentence in a LiveKit room using Microsoft AI TTS, without STT or an LLM.

Run with ``dev`` and connect a playback-enabled, subscribe-only participant.
Set LIVEKIT_URL/API_KEY/API_SECRET for the server and MICROSOFT_AI_ENV_FILE for
the private TTS configuration. A local LiveKit server needs no Cloud account.
"""

import asyncio
import logging
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    cli,
    room_io,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import microsoft_ai

GREETING = "Hello, this is a Microsoft AI voice test."
logger = logging.getLogger("microsoft-ai-tts-room")
server = AgentServer(host="127.0.0.1")


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Publish one greeting after a participant has subscribed, then release the session."""
    async with microsoft_ai.TTS(env_file=os.environ.get("MICROSOFT_AI_ENV_FILE")) as speech:
        session: AgentSession = AgentSession(
            tts=speech,
            vad=None,
            turn_handling={"turn_detection": None},
            user_away_timeout=None,
            conn_options=SessionConnectOptions(
                tts_conn_options=APIConnectOptions(max_retry=0, timeout=10.0)
            ),
        )
        try:
            await session.start(
                agent=Agent(instructions="Speak only the supplied greeting."),
                room=ctx.room,
                room_options=room_io.RoomOptions(
                    audio_input=False,
                    video_input=False,
                    text_input=False,
                    audio_output=room_io.AudioOutputOptions(sample_rate=speech.sample_rate),
                    text_output=False,
                ),
                session_host=False,
                record=False,
            )
            await asyncio.wait_for(session.room_io.wait_for_ready(), timeout=30.0)
            handle = session.say(GREETING, allow_interruptions=False, add_to_chat_ctx=False)
            await asyncio.wait_for(handle, timeout=45.0)
            # Awaiting a SpeechHandle completes even on synthesis failure.
            if error := handle.exception():
                raise error
            logger.info("The one-shot TTS greeting finished playing to the room")
        finally:
            await session.aclose()


if __name__ == "__main__":
    cli.run_app(server)
