from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
import asyncio
import logging
import os
import signal

from livekit import rtc
from livekit.plugins.pinch import Translator, TranslatorOptions, TranscriptEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration — read from environment

LIVEKIT_URL: str = os.environ["LIVEKIT_URL"]
LIVEKIT_TOKEN: str = os.environ["LIVEKIT_TOKEN"]

# Optionally override language settings via environment variables.
SOURCE_LANGUAGE: str = os.environ.get("SOURCE_LANGUAGE", "en-US")
TARGET_LANGUAGE: str = os.environ.get("TARGET_LANGUAGE", "es-ES")
VOICE_TYPE: str = os.environ.get("VOICE_TYPE", "clone")

# Transcript handler

def on_transcript(event: TranscriptEvent) -> None:
    """Called for every transcript message received from the Pinch data channel."""
    status = "✓ final" if event.is_final else "… interim"
    kind = "ORIGINAL  " if event.is_original else "TRANSLATED"
    print(
        f"[{kind}] [{event.language_detected}] [{status}]  {event.text}",
        flush=True,
    )

# Main


async def main() -> None:
    # Connect to the LiveKit room
    room = rtc.Room()

    @room.on("disconnected")
    def on_disconnected(reason: str) -> None:
        logger.warning("Room disconnected: %s", reason)

    @room.on("participant_connected")
    def on_participant(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant connected: %s (%s)", participant.name, participant.sid)

    logger.info("Connecting to LiveKit  url=%s", LIVEKIT_URL)
    await room.connect(LIVEKIT_URL, LIVEKIT_TOKEN)
    logger.info(
        "Connected to room=%s  local_participant=%s",
        room.name,
        room.local_participant.sid,
    )

    # Create and configure the Pinch Translator
    translator = Translator(
        options=TranslatorOptions(
            source_language=SOURCE_LANGUAGE,
            target_language=TARGET_LANGUAGE,
            voice_type=VOICE_TYPE,
        )
        # api_key is read from PINCH_API_KEY env var automatically
    )

    # Register the transcript callback (decorator style also works, see docs).
    translator.on_transcript(on_transcript)

    # Start the translation pipeline
    logger.info(
        "Starting Pinch translation  %s → %s  voice=%s",
        SOURCE_LANGUAGE,
        TARGET_LANGUAGE,
        VOICE_TYPE,
    )
    await translator.start(room)
    logger.info(
        "Translation active. Speak into your microphone. "
        'Press Ctrl+C to stop.\n'
        "Translated audio is being published back to the LiveKit room "
        "as track 'pinch-translated'.\n"
    )

    # Keep running until the user hits Ctrl+C
    stop_event = asyncio.Event()

    def _signal_handler(*_):
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    # Clean shutdown
    logger.info("Shutting down…")
    await translator.stop()
    await room.disconnect()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())