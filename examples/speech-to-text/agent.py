import asyncio
import logging

from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.plugins.deepgram import STT


async def entrypoint(job: JobContext):
    logging.info("starting tts example agent")
    stt = STT()
    stt_stream = stt.stream()

    audio_stream_future = asyncio.Future[rtc.AudioStream]()

    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream_future.set_result(rtc.AudioStream(track))

    job.room.on("track_subscribed", on_track_subscribed)

    for participant in job.room.participants.values():
        for track_pub in participant.tracks.values():
            # This track is not yet subscribed, when it is subscribed it will
            # call the on_track_subscribed callback
            if track_pub.track is None:
                continue
            audio_stream_future.set_result(rtc.AudioStream(track_pub.track))

    async def process_track(audio_stream: rtc.AudioStream):
        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)

    async def process_stt():
        async for stt_event in stt_stream:
            if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                logging.info("Got transcript: %s", stt_event.alternatives[0].text)

    audio_stream = await audio_stream_future
    await asyncio.gather(process_track(audio_stream), process_stt())


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
