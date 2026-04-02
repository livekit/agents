"""Comparison voice agent using Mistral STT and ElevenLabs TTS.

Requires:
    MISTRAL_API_KEY for Mistral STT
    OPENAI_API_KEY for the LLM
    ELEVEN_API_KEY for ElevenLabs TTS

This example uses Mistral's realtime streaming STT model, OpenAI for the LLM,
and ElevenLabs Flash v2.5 for TTS using an explicit voice ID already referenced
elsewhere in the repo. The Mistral STT settings are tuned to the best plugin-only
configuration we observed in comparison runs against the Deepgram example.

For the cleanest console-mode testing, use headphones to avoid the agent hearing
its own playback through the microphone.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    room_io,
)
from livekit.plugins import elevenlabs, mistralai, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("mistral-realtime-stt-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly, built by LiveKit. "
                "Keep responses concise, natural, and voice-friendly. "
                "Reply in one or two short sentences."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the user and mention that you are using Mistral for speech-to-text "
                "and ElevenLabs for text-to-speech. Keep the greeting short."
            )
        )


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=mistralai.STT(
            model="voxtral-mini-transcribe-realtime-2602",
            target_streaming_delay_ms=240,
            chunk_duration_ms=10,
            finalize_delay_ms=650,
        ),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=elevenlabs.TTS(
            model="eleven_flash_v2_5",
            voice_id="hpp4J3VqNfWAUOO0d1Us",
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.0,
            },
        ),
        preemptive_generation=False,
        aec_warmup_duration=3.0,
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
