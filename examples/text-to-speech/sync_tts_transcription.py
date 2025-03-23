import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    transcription,
    tts,
)
from livekit.plugins import elevenlabs

load_dotenv()

logger = logging.getLogger("transcription-forwarding-demo")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting transcription protocol example")
    tts_11labs = elevenlabs.TTS()

    # publish an audio track
    source = rtc.AudioSource(tts_11labs.sample_rate, tts_11labs.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await ctx.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    # start the transcription examples
    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=ctx.room, participant=ctx.room.local_participant
    )

    await _eg_single_segment(tts_forwarder, tts_11labs, source)

    await asyncio.sleep(2)
    await _eg_streamed_tts_stream(tts_forwarder, tts_11labs, source)


async def _eg_single_segment(
    tts_forwarder: transcription.TTSSegmentsForwarder,
    tts_11labs: tts.TTS,
    source: rtc.AudioSource,
):
    """Transcription example without streaming (single string)"""

    text = "Hello world, this is a single segment"
    logger.info("pushing text %s", text)
    tts_forwarder.push_text(text)
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()
    playout_task = asyncio.create_task(_playout_task(tts_forwarder, playout_q, source))

    async for output in tts_11labs.synthesize(text):
        tts_forwarder.push_audio(output.frame)
        playout_q.put_nowait(output.frame)

    tts_forwarder.mark_audio_segment_end()
    playout_q.put_nowait(None)

    await playout_task


async def _eg_streamed_tts_stream(
    tts_forwarder: transcription.TTSSegmentsForwarder,
    tts_11labs: tts.TTS,
    source: rtc.AudioSource,
):
    """Transcription example using a tts stream (we split text into chunks just for the example)"""

    # this tts_forwarder will forward the transcription to the client and sync with the audio
    tts_stream = tts_11labs.stream()

    streamed_text = "Hello world, this text is going to be splitted into small chunks"
    logger.info("pushing text %s", streamed_text)
    for chunk in _text_to_chunks(streamed_text):
        tts_stream.push_text(chunk)
        tts_forwarder.push_text(chunk)

    tts_stream.flush()
    tts_stream.end_input()
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async def _synth_task() -> None:
        async for ev in tts_stream:
            playout_q.put_nowait(ev.frame)
            tts_forwarder.push_audio(ev.frame)

        tts_forwarder.mark_audio_segment_end()
        playout_q.put_nowait(None)
        await tts_stream.aclose()

    playout_task = asyncio.create_task(_playout_task(tts_forwarder, playout_q, source))
    synth_task = asyncio.create_task(_synth_task())

    await asyncio.gather(synth_task, playout_task)

    await tts_forwarder.aclose()


async def _playout_task(
    tts_forwarder: transcription.TTSSegmentsForwarder,
    playout_q: asyncio.Queue,
    audio_source: rtc.AudioSource,
) -> None:
    """Playout audio frames from the queue to the audio source"""
    tts_forwarder.segment_playout_started()
    while True:
        frame = await playout_q.get()
        if frame is None:
            break

        await audio_source.capture_frame(frame)

    tts_forwarder.segment_playout_finished()


def _text_to_chunks(text: str) -> list[str]:
    """Split the text into chunks of 2, 3, and 4 words"""
    sizes = [2, 3, 4]
    chunks, i = [], 0

    for size in sizes:
        while i + size <= len(text):
            chunks.append(text[i : i + size])
            i += size

    chunks.append(text[i:])  # remaining
    return chunks


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
