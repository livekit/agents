import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioTurnContext,
    AudioTurnDetector,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.agents.utils.audio import calculate_audio_duration
from livekit.plugins import silero

logger = logging.getLogger("audio-turn-detector")
logger.setLevel(logging.INFO)

load_dotenv()


class HeuristicAudioTurnDetector(AudioTurnDetector):
    """Example audio-native turn detector.

    This intentionally uses a simple duration heuristic so the example stays dependency-free.
    A production detector can inspect ``turn_ctx.audio`` and run any external model there.
    """

    def __init__(
        self,
        *,
        short_utterance_seconds: float = 0.8,
        long_utterance_seconds: float = 1.8,
    ) -> None:
        self._short_utterance_seconds = short_utterance_seconds
        self._long_utterance_seconds = long_utterance_seconds

    @property
    def model(self) -> str:
        return "heuristic-audio-turn-detector"

    @property
    def provider(self) -> str:
        return "examples"

    async def unlikely_threshold(self, language) -> float | None:
        return 0.5

    async def supports_language(self, language) -> bool:
        return True

    async def predict_end_of_turn_audio(
        self, turn_ctx: AudioTurnContext, *, timeout: float | None = None
    ) -> float:
        duration = calculate_audio_duration(turn_ctx.audio)
        logger.info(
            "audio turn detector evaluated current turn",
            extra={
                "duration_seconds": round(duration, 3),
                "language": turn_ctx.language,
                "transcript": turn_ctx.transcript,
            },
        )

        if not turn_ctx.transcript.strip():
            return 0.0
        if duration <= self._short_utterance_seconds:
            return 0.95
        if duration >= self._long_utterance_seconds:
            return 0.15
        return 0.55


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=HeuristicAudioTurnDetector(),
            endpointing={
                "min_delay": 0.05,
                "max_delay": 0.6,
            },
        ),
    )

    await session.start(
        agent=Agent(
            instructions=("You are a concise voice assistant. Reply briefly and naturally.")
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
