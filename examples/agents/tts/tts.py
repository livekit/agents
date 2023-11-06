import json
import asyncio
from livekit import rtc
from livekit.plugins.vad import VAD, VADPlugin
from livekit.plugins.elevenlabs import ElevenLabsTTSPlugin
from livekit import agents
from typing import AsyncIterator

SAMPLE_RATE = 48000
NUM_CHANNELS = 1


async def tts_agent(ctx: agents.JobContext):
    tts = ElevenLabsTTSPlugin()
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("echo", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await ctx.room.local_participant.publish_track(track, options)

    @ctx.room.on("data_received")
    def on_data_received(data: bytes, participant: rtc.RemoteParticipant, kind):
        payload = json.loads(data.decode('utf-8'))
        if payload["type"] != "tts":
            return

        text = payload["text"]
        iterator = agents.utils.AsyncIteratorList([text])
        tts.push(iterator)
        print(f"Data Received: {text}")

    # async for response_frames in tts.stream():
    #     print("RFrame")
    #     for frame in response_frames:
    #         print("Frame")
    #         resampled = frame.remix_and_resample(
    #             SAMPLE_RATE, NUM_CHANNELS)
    #         await source.capture_frame(resampled)
