"""Minimal voice agent example using Mistral for speech input and output.

Requires:
    MISTRAL_API_KEY for Mistral STT/TTS
    OPENAI_API_KEY for the LLM

This example uses Mistral's realtime streaming STT model and Voxtral TTS, while
keeping OpenAI as the LLM because we could not verify an official recommended
full-stack Mistral voice-agent example in the published docs.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    TurnHandlingOptions,
    cli,
    metrics,
    room_io,
)
from livekit.plugins import mistralai, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("mistral-realtime-stt-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly, built by LiveKit. "
                "Keep responses concise, natural, and voice-friendly."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the user and mention that you are using Mistral for speech-to-text "
                "and text-to-speech."
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
        stt=mistralai.STT(model="voxtral-mini-transcribe-realtime-2602"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=mistralai.TTS(voice="en_paul_neutral"),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.0,
            },
        ),
        preemptive_generation=True,
        aec_warmup_duration=3.0,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
