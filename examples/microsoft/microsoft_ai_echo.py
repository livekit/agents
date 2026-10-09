"""Microphone -> Microsoft AI STT -> Microsoft AI TTS, without an LLM.

Run with ``dev`` using a local LiveKit server and an explicitly started browser
microphone. MICROSOFT_AI_ENV_FILE selects the private STT/TTS configuration.
Microphone audio goes to STT; final text goes to TTS and transient room captions.
Use headphones. This example neither records audio nor logs transcripts.
"""

import asyncio
import logging
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    CloseEvent,
    JobContext,
    StopResponse,
    cli,
    inference,
    llm,
    room_io,
)
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import microsoft_ai

logger = logging.getLogger("microsoft-ai-echo")
server = AgentServer(host="127.0.0.1")
SESSION_LIMIT = 180.0
VAD_SILENCE = 0.5


class EchoAgent(Agent):
    """Echo completed user turns through normal, interruptible session speech."""

    def __init__(self) -> None:
        super().__init__(instructions="Echo the finalized user speech without an LLM.")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if text := new_message.text_content:
            handle = self.session.say(text, allow_interruptions=True, add_to_chat_ctx=False)
            handle.add_done_callback(self._speech_done)
        raise StopResponse()

    @staticmethod
    def _speech_done(handle: SpeechHandle) -> None:
        if error := handle.exception():
            logger.error("Echo synthesis failed (%s)", type(error).__name__)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one bounded echo session and release its providers when the user leaves."""
    detector = inference.VAD(min_silence_duration=VAD_SILENCE)
    env_file = os.environ.get("MICROSOFT_AI_ENV_FILE")
    # Native STT needs its own VAD stream to commit; AgentSession's VAD does not do that.
    async with (
        microsoft_ai.STT(vad=detector, env_file=env_file) as recognizer,
        microsoft_ai.TTS(env_file=env_file) as speech,
    ):
        session: AgentSession = AgentSession(
            stt=recognizer,
            tts=speech,
            vad=detector,
            turn_handling={
                "turn_detection": "stt",
                "interruption": {
                    "enabled": True,
                    "mode": "vad",
                    "min_duration": 0.2,
                    "min_words": 0,
                    "resume_false_interruption": False,
                },
                "preemptive_generation": {"enabled": False},
            },
            aec_warmup_duration=None,
            user_away_timeout=None,
            conn_options=SessionConnectOptions(
                stt_conn_options=APIConnectOptions(max_retry=0, timeout=10.0),
                tts_conn_options=APIConnectOptions(max_retry=0, timeout=10.0),
                max_unrecoverable_errors=0,
            ),
        )
        closed = asyncio.Event()

        @session.on("close")
        def on_close(_: CloseEvent) -> None:
            closed.set()

        try:
            await session.start(
                agent=EchoAgent(),
                room=ctx.room,
                room_options=room_io.RoomOptions(
                    audio_input=room_io.AudioInputOptions(
                        sample_rate=16000, pre_connect_audio=False
                    ),
                    video_input=False,
                    text_input=False,
                    audio_output=room_io.AudioOutputOptions(sample_rate=speech.sample_rate),
                    text_output=True,
                ),
                session_host=False,
                record=False,
            )
            await asyncio.wait_for(session.room_io.wait_for_ready(), timeout=10.0)
            try:
                await asyncio.wait_for(closed.wait(), timeout=SESSION_LIMIT)
            except asyncio.TimeoutError:
                logger.info("Echo demo reached its three-minute session limit")
        finally:
            await session.aclose()
    ctx.shutdown(reason="Echo session ended")


if __name__ == "__main__":
    cli.run_app(server)
