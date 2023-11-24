# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import asyncio
from livekit import rtc
from livekit.plugins.vad import VAD, VADPlugin
from livekit.plugins.elevenlabs import ElevenLabsTTSPlugin
from livekit import agents

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
    def on_data_received(
            data: bytes,
            participant: rtc.RemoteParticipant,
            kind):
        payload = json.loads(data.decode('utf-8'))
        if payload["type"] != "tts":
            return

        text = payload["text"]
        iterator = tts.process(agents.utils.AsyncIteratorList([text]))

        async def speak():
            async for frame_iter in iterator:
                async for frame in frame_iter:
                    resampled = frame.remix_and_resample(
                        SAMPLE_RATE, NUM_CHANNELS)
                    await source.capture_frame(resampled)

        asyncio.create_task(speak())
