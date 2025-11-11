from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from vad_extension import AgentSpeechState, VADExtension, VADExtensionConfig

load_dotenv(".env.local")

logger = logging.getLogger(__name__)

DEFAULT_FILLER_WORDS = [
    "uh",
    "um",
    "umm",
    "er",
    "ah",
    "oh",
    "hmm",
    "huh",
    "haan",
    "haanji",
    "haina",
    "achha",
    "arey",
]


def _load_ignored_words() -> list[str]:
    env_value = os.getenv("LIVEKIT_IGNORED_FILLERS")
    if env_value is None:
        return DEFAULT_FILLER_WORDS.copy()

    values = [word.strip() for word in env_value.split(",") if word.strip()]
    return values if values else DEFAULT_FILLER_WORDS.copy()


def _load_decision_timeout() -> float:
    raw = os.getenv("LIVEKIT_DECISION_TIMEOUT")
    if not raw:
        return 0.7
    try:
        return max(0.1, float(raw))
    except ValueError:
        logger.warning("Invalid LIVEKIT_DECISION_TIMEOUT value '%s', falling back to 0.7s", raw)
        return 0.7


IGNORED_FILLER_WORDS = _load_ignored_words()
DECISION_TIMEOUT = _load_decision_timeout()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )


def _create_vad_extension() -> VADExtension:
    base_vad = silero.VAD.load()
    extension = VADExtension(
        base_vad,
        config=VADExtensionConfig(
            decision_timeout=DECISION_TIMEOUT,
            ignored_words=IGNORED_FILLER_WORDS.copy(),
        ),
    )
    return extension


def _normalize_transcript(text: str) -> str:
    words = []
    for token in text.split():
        normalized = token.strip(".,!?").lower()
        if normalized in IGNORED_FILLER_WORDS:
            continue
        words.append(token)
    return " ".join(words)


def _register_transcription_hooks(session: AgentSession) -> None:
    @session.on("user_input_transcribed")
    def _on_transcribed(ev) -> None:
        if not getattr(ev, "is_final", False):
            return

        transcript = getattr(ev, "transcript", "")
        if not transcript:
            return

        normalized = _normalize_transcript(transcript)
        logger.info(
            "User utterance transcribed",
            extra={
                "raw_transcript": transcript,
                "normalized_transcript": normalized,
            },
        )


def _register_agent_state_hooks(session: AgentSession, vad_extension: VADExtension) -> None:
    @session.on("agent_state_changed")
    def _on_agent_state_changed(event) -> None:
        new_state = getattr(event, "new_state", None)
        if new_state == "speaking":
            vad_extension.set_agent_state(AgentSpeechState.SPEAKING)
        else:
            vad_extension.set_agent_state(AgentSpeechState.IDLE)


def _register_interruption_handler(session: AgentSession, vad_extension: VADExtension) -> None:
    async def _on_interruption(transcript: str) -> None:
        try:
            logger.info("Pausing agent due to interruption transcript: '%s'", transcript)
            interrupt_future = session.interrupt(force=True)
            await interrupt_future
        except Exception:
            logger.exception("Failed to interrupt session after user interruption.")

    vad_extension.set_interruption_handler(_on_interruption)


async def entrypoint(ctx: agents.JobContext):
    vad_extension = _create_vad_extension()
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=vad_extension,
        turn_detection=MultilingualModel(),
    )

    stt_instance = getattr(session, "_stt", None)
    if stt_instance is not None:
        vad_extension.set_asr_stream_factory(lambda: stt_instance.stream())

    _register_agent_state_hooks(session, vad_extension)
    _register_interruption_handler(session, vad_extension)
    _register_transcription_hooks(session)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    