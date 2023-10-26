import asyncio
import livekit.rtc as rtc
from livekit.processors.vad import VAD
from typing import AsyncIterator, Callable


def vad_agent(room: rtc.Room, participant: rtc.RemoteParticipant):

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad = VAD(silence_threshold_ms=250)

        async def vad_result_loop(self, queue: AsyncIterator[VAD.Event]):
            async for event in queue:
                if event.type == "voice_started":
                    print("VAD - Voice Started")
                elif event.type == "voice_finished":
                    print(f"VAD - Voice Finished. Frame Count: {len(event.frames)}")

        asyncio.create_task(vad_result_loop(vad.stream()))

        async for frame in audio_stream:
            vad.push_frame(frame)

    @room.on("track_available")
    def on_track_available(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind == rtc.TrackKind.AUDIO:
            publication.set_subscribed(True)

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.AUDIO:
            return

        asyncio.create_task(process_track(track))
