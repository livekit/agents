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

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)

    def on_track_subscribed(
        track: rtc.Track,
        pub: rtc.TrackPublication,
        rp: rtc.RemoteParticipant,
    ):
        logging.info("NEILLLL subscribed to track %s", track.sid)
        asyncio.create_task(process_track(track))

    job.room.on("track_subscribed", on_track_subscribed)

    async for stt_event in stt_stream:
        if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
            logging.info("Got transcript: %s", stt_event.alternatives[0].text)


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
