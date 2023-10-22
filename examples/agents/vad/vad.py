import asyncio
from typing import AsyncIterator
from livekit.rtc as rtc
from livekit.agents import Agent, Processor
from livekit.processors.vad import VAD


class VADAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vad = VAD(silence_threshold_ms=250)

    def on_audio_track(self, track: rtc.Track, participant: rtc.Participant):
        loop = asyncio.get_event_loop()
        loop.create_task(self.process(track))

    async def process(self, participant: rtc.Participant, audio_track: rtc.Track):
        audio_stream = rtc.AudioStream(audio_track)
        vad_processor = Processor(process=self.vad.push_frame)
        asyncio.create_task(self.vad_result_loop(vad_processor.stream()))
        async for frame in audio_stream:
            vad_processor.push(frame)

    async def vad_result_loop(self, participant: rtc.Participant, queue: AsyncIterator[VAD.Event]):
        async for event in queue:
            if event.type == "voice_started":
                await self.participant.publish_data(f"{participant.identity} started talking")
            elif event.type == "voice_finished":
                await self.participant.publish_data(f"{participant.identity} stopped talking")

    def should_process(self, track: rtc.TrackPublication, participant: rtc.Participant) -> bool:
        return track.kind == rtc.TrackKind.Audio