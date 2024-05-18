import asyncio
import logging

from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram


async def _log_transcription(stt_stream: agents.stt.SpeechStream):
    async for ev in stt_stream:
        if ev.type == agents.stt.SpeechEventType.INTERIM_TRANSCRIPT:
            print(ev.alternatives[0].text, end="")
        elif ev.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
            print(" -> ", ev.alternatives[0].text)


async def entrypoint(job: JobContext):
    logging.info("starting deepgram tts example")
    deepgram_stt = deepgram.STT()

    tasks = []

    async def transcribe_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)

        stt_stream = deepgram_stt.stream()
        stt_task = asyncio.create_task(_log_transcription(stt_stream))
        tasks.append(stt_task)

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @job.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            tasks.append(asyncio.create_task(transcribe_track(track)))


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
