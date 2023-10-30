import asyncio
import livekit.rtc as rtc
from livekit.processors.vad import VAD, VADProcessor
from livekit import agents
from typing import AsyncIterator


async def vad_agent(params: agents.Job.AgentParams):

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad_processor = VADProcessor(silence_threshold_ms=250)

        async def vad_result_loop(queue: AsyncIterator[VAD.Event]):
            async for event in queue:
                if event.type == "voice_started":
                    print("VAD - Voice Started")
                elif event.type == "voice_finished":
                    asyncio.create_task(params.room.local_participant.publish_data(
                        f"Voice Detected For: {len(event.frames) * 10.0 / 1000.0} seconds"))
                    print(
                        f"VAD - Voice Finished. Frame Count: {len(event.frames)}")

        asyncio.create_task(vad_result_loop(queue=vad_processor.stream()))

        async for frame in audio_stream:
            vad_processor.push(frame)

    @params.room.on("track_available")
    def on_track_available(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind == rtc.TrackKind.KIND_AUDIO:
            publication.set_subscribed(True)

    @params.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(track))
