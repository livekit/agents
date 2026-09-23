"""Live speech-to-speech translation with Camb.ai realtime.

Every participant who publishes audio gets a translated track published back into the
room, named ``translated-<target language>``, plus the source transcript and the
translated text on the console. One websocket per speaker does the whole job: no STT,
no LLM, no TTS.

Run it with a Camb.ai key and LiveKit credentials in the environment:

    export CAMB_API_KEY=...
    export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
    python camb_realtime_translator.py dev

The realtime endpoint speaks 24 kHz mono PCM16; room audio is resampled for you.
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.plugins import cambai

load_dotenv()

logger = logging.getLogger("camb-translator")

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "fr-FR"

# "fast" begins speaking sooner; "slow" covers a longer language list. Both translate
# every complete utterance, so a live translator wants the lower latency.
MODE = "fast"

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# The event loop only holds a weak reference to a task, so keep them alive here.
_tasks: set[asyncio.Task[None]] = set()


def _spawn(coro: asyncio.coroutines) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def translate_track(ctx: JobContext, track: rtc.Track, identity: str) -> None:
    """Translate one participant's audio and publish the result as its own track."""
    model = cambai.experimental.realtime.RealtimeModel(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        mode=MODE,
    )
    session = model.session()

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    publication = await ctx.room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track(f"translated-{TARGET_LANGUAGE}", source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("translating %s into %s on %s", identity, TARGET_LANGUAGE, publication.sid)

    @session.on("input_audio_transcription_completed")
    def _on_transcript(ev: object) -> None:
        logger.info("%s said: %s", identity, ev.transcript)  # type: ignore[attr-defined]

    @session.on("error")
    def _on_error(ev: object) -> None:
        logger.error("translation failed for %s: %s", identity, ev.error)  # type: ignore[attr-defined]

    @session.on("generation_created")
    def _on_generation(ev: object) -> None:
        async def show_text(message: object) -> None:
            text = ""
            async for chunk in message.text_stream:  # type: ignore[attr-defined]
                text += chunk
            if text:
                logger.info("%s translated: %s", identity, text)

        async def play_audio(message: object) -> None:
            async for frame in message.audio_stream:  # type: ignore[attr-defined]
                await source.capture_frame(frame)

        async def deliver() -> None:
            async for msg in ev.message_stream:  # type: ignore[attr-defined]
                await asyncio.gather(show_text(msg), play_audio(msg))

        _spawn(deliver())

    try:
        async for event in rtc.AudioStream(track):
            session.push_audio(event.frame)
    finally:
        await session.aclose()
        await model.aclose()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    @ctx.room.on("track_subscribed")
    def _on_track(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            _spawn(translate_track(ctx, track, participant.identity))


if __name__ == "__main__":
    cli.run_app(server)
