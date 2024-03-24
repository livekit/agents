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
import logging
from typing import Optional, Set

from livekit import agents, rtc
from livekit.plugins.silero import VADEventType, VADPlugin


class VAD:
    def __init__(self):
        # plugins
        self.vad_plugin = VADPlugin(left_padding_ms=1000, silence_threshold_ms=500)

        self.ctx: Optional[agents.JobContext] = None
        self.track_tasks: Set[asyncio.Task] = set()
        self.line_out: Optional[rtc.AudioSource] = None

    async def start(self, ctx: agents.JobContext):
        self.ctx = ctx
        ctx.room.on("track_subscribed", self.on_track_subscribed)
        ctx.room.on("disconnected", self.cleanup)
        await self.publish_audio()

    async def publish_audio(self):
        self.line_out = rtc.AudioSource(48000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.line_out)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.ctx.room.local_participant.publish_track(track, options)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    def cleanup(self):
        logging.info("VAD agent clean up")

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        async for vad_result in self.vad_plugin.start(audio_stream):
            if vad_result.type == VADEventType.STARTED:
                pass
            elif vad_result.type == VADEventType.FINISHED:
                for frame in vad_result.frames:
                    await self.line_out.capture_frame(frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("VAD agent received job request")
        vad = VAD()

        await job_request.accept(
            vad.start,
            auto_subscribe=agents.AutoSubsribe.AUDIO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
            identity="vad_agent",
            name="VAD",
        )

    worker = agents.Worker(
        request_handler=job_request_cb, worker_type=agents.JobType.JT_ROOM
    )
    agents.run_app(worker)
""
