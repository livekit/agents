import asyncio
import livekit.rtc as rtc
from livekit import agents
from livekit.processors.vad import VADProcessor, VAD
from livekit.processors.openai import WhisperOpenSourceTranscriberProcessor
from typing import AsyncIterator


async def stt_agent(ctx: agents.JobContext):

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad_processor = VADProcessor(
            left_padding_ms=250, silence_threshold_ms=500)
        stt_processor = WhisperOpenSourceTranscriberProcessor()

        vad_results = vad_processor\
            .start(audio_stream)\
            .filter(lambda data: data.type == agents.VADProcessorEventType.FINISHED)\
            .map(lambda data: data.frames)\
            .unwrap()
        stt_results = stt_processor.start(vad_results)

        print("NEIL stt_results:", stt_results)

        async for event in stt_results:
            print("NEIL event", event)
            if event.type == agents.ProcessorEventType.ERROR:
                continue

            text = event.data.text
            asyncio.create_task(ctx.room.local_participant.publish_data(text))

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(track))
