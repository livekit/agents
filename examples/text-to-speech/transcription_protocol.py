import asyncio
import logging
import os
from typing import Optional

from livekit import rtc
from livekit.agents import (
    JobContext,
    JobRequest,
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


async def entrypoint(job: JobContext):
    logging.info("starting transcription protocol example")

    tts_11labs = elevenlabs.TTS()

    source = rtc.AudioSource(tts_11labs.sample_rate, tts_11labs.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await job.room.local_participant.publish_track(track, options)



    #
    # 1. example using a tts stream (we split text into chunks just for the example)
    #
    await asyncio.sleep(2)

    # this tts_forwarder will forward the transcription to the client and sync with the audio
    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=job.room, participant=job.room.local_participant
    )

    tts_stream = tts_11labs.stream()
    streamed_text = "Hello world, this text is going to be splitted into small chunks"
    logging.info("pushing text %s", streamed_text)
    for chunk in _text_to_chunks(streamed_text):
        tts_stream.push_text(chunk)
        tts_forwarder.push_text(chunk)

    tts_stream.mark_segment_end()
    tts_forwarder.mark_text_segment_end()

    second_streamed_text = "This is another segment that will be streamed"
    logging.info("pushing text %s", second_streamed_text)
    for chunk in _text_to_chunks(second_streamed_text):
        tts_stream.push_text(chunk)
        tts_forwarder.push_text(chunk)

    tts_stream.mark_segment_end()
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async def _synth_task():
        async for ev in tts_stream:
            if ev.type != tts.SynthesisEventType.AUDIO:
                continue
            assert ev.audio is not None
            playout_q.put_nowait(ev.audio.data)

        playout_q.put_nowait(None)

    synth_task = asyncio.create_task(_synth_task())
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    await tts_stream.aclose(wait=True)
    await asyncio.gather(synth_task, playout_task)
    await tts_forwarder.aclose()

    #
    # 2. example without streaming (single segment)
    #
    await asyncio.sleep(2)

    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=job.room, participant=job.room.local_participant
    )

    text = "Hello world, this is a single segment"
    logging.info("pushing text %s", text)
    tts_forwarder.push_text(text)
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    async for output in tts_11labs.synthesize(text):
        tts_forwarder.push_audio(output.data)
        playout_q.put_nowait(output.data)

    tts_forwarder.mark_audio_segment_end()

    await tts_forwarder.aclose()

    #
    # 3. example with deferred playout
    #
    await asyncio.sleep(2)
    
    # disable auto playout
    tts_forwarder = transcription.TTSSegmentsForwarder(
        room=job.room,
        participant=job.room.local_participant,
        auto_playout=False,
    )

    text = "Hello world, this is a single segment with deferred playout"
    logging.info("pushing text %s", text)
    tts_forwarder.push_text(text)
    tts_forwarder.mark_text_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async for output in tts_11labs.synthesize(text):
        tts_forwarder.push_audio(output.data)
        playout_q.put_nowait(output.data)

    tts_forwarder.mark_audio_segment_end()

    await asyncio.sleep(2)
    logging.info("waiting 2 seconds before starting playout")

    tts_forwarder.segment_playout_started()
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    playout_q.put_nowait(None)
    await playout_task

    await tts_forwarder.aclose()


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
