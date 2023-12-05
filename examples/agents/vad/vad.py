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

import asyncio
import livekit.rtc as rtc
from livekit.plugins import core
from livekit.plugins.vad import VADPlugin
from livekit import agents

SAMPLE_RATE = 48000
NUM_CHANNELS = 1


async def vad_agent(ctx: agents.JobContext):

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("echo", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await ctx.room.local_participant.publish_track(track, options)

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad_plugin = VADPlugin(
            left_padding_ms=200, silence_threshold_ms=250)

        vad_results = vad_plugin.start(audio_stream) .filter(
            lambda data: data.type == core.VADPluginResultType.FINISHED) .map(
            lambda data: data.frames) .unwrap()

        async for frames in vad_results:
            asyncio.create_task(ctx.room.local_participant.publish_data(
                f"Voice Detected For: {len(frames) * 10.0 / 1000.0} seconds"))
            for frame in frames:
                resampled = frame.remix_and_resample(
                    SAMPLE_RATE, NUM_CHANNELS)
                await source.capture_frame(resampled)
            print(
                f"VAD - Voice Finished. Frame Count: {len(frames)}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO or track.name == "echo":
            return
        asyncio.create_task(process_track(track))
