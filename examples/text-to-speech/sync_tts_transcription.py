import asyncio
import logging
from typing import Optional

from livekit import rtc
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    transcription,
    tts,
)
from livekit.plugins import elevenlabs


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


async def _playout_task(
    playout_q: asyncio.Queue, audio_source: rtc.AudioSource
) -> None:
    """Playout audio frames from the queue to the audio source"""
    while True:
        frame = await playout_q.get()
        if frame is None:
            break

        await audio_source.capture_frame(frame)


async def _eg_streamed_tts_stream(
    ctx: JobContext, tts_11labs: tts.TTS, source: rtc.AudioSource
):
    """Transcription example using a tts stream (we split text into chunks just for the example)"""

    # this tts_forwarder will forward the transcription to the client and sync with the audio
    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=ctx.room, participant=ctx.room.local_participant
    )

    tts_stream = tts_11labs.stream()

    streamed_text = "Hello world, this text is going to be splitted into small chunks"
    logging.info("pushing text %s", streamed_text)
    for chunk in _text_to_chunks(streamed_text):
        tts_stream.push_text(chunk)
        tts_forwarder.push_text(chunk)

    tts_stream.flush()
    tts_forwarder.mark_text_segment_end()

    second_streamed_text = "This is another segment that will be streamed"
    logging.info("pushing text %s", second_streamed_text)
    for chunk in _text_to_chunks(second_streamed_text):
        tts_stream.push_text(chunk)
        tts_forwarder.push_text(chunk)

    tts_stream.flush()
    tts_stream.end_input()
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async def _synth_task():
        async for ev in tts_stream:
            playout_q.put_nowait(ev.frame)

        playout_q.put_nowait(None)

    synth_task = asyncio.create_task(_synth_task())
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    await asyncio.gather(synth_task, playout_task)
    await tts_stream.aclose()
    await tts_forwarder.aclose()


async def _eg_single_segment(
    ctx: JobContext, tts_11labs: tts.TTS, source: rtc.AudioSource
):
    """Transcription example without streaming (single segment"""

    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=ctx.room, participant=ctx.room.local_participant
    )

    text = "Hello world, this is a single segment"
    logging.info("pushing text %s", text)
    tts_forwarder.push_text(text)
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    async for output in tts_11labs.synthesize(text):
        tts_forwarder.push_audio(output.frame)
        playout_q.put_nowait(output.frame)

    tts_forwarder.mark_audio_segment_end()
    playout_q.put_nowait(None)

    await tts_forwarder.aclose()
    await playout_task


async def _eg_deferred_playout(
    ctx: JobContext, tts_11labs: tts.TTS, source: rtc.AudioSource
):
    """example with deferred playout (We have a synthesized audio before starting to play it)"""
    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=ctx.room, participant=ctx.room.local_participant
    )

    text = "Hello world, this is a single segment with deferred playout"
    logging.info("pushing text %s", text)
    tts_forwarder.push_text(text)
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async for output in tts_11labs.synthesize(text):
        tts_forwarder.push_audio(output.frame)
        playout_q.put_nowait(output.frame)

    tts_forwarder.mark_audio_segment_end()

    logging.info("waiting 2 seconds before starting playout")
    await asyncio.sleep(2)

    tts_forwarder.segment_playout_started()
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    playout_q.put_nowait(None)
    tts_forwarder.segment_playout_finished()
    await playout_task
    await tts_forwarder.aclose()


async def entrypoint(ctx: JobContext):
    logging.info("starting transcription protocol example")

    tts_11labs = elevenlabs.TTS()

    # publish an audio track
    source = rtc.AudioSource(tts_11labs.sample_rate, tts_11labs.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    await ctx.connect()
    await ctx.room.local_participant.publish_track(track, options)

    # start the transcription examples
    await asyncio.sleep(2)
    await _eg_streamed_tts_stream(ctx, tts_11labs, source)
    await asyncio.sleep(2)
    await _eg_single_segment(ctx, tts_11labs, source)
    await asyncio.sleep(2)
    await _eg_deferred_playout(ctx, tts_11labs, source)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
