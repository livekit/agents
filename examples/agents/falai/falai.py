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
import logging

from livekit import rtc, agents
from livekit.plugins.fal import SDXLPlugin


class FalAI:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        falai = FalAI(ctx)
        await falai.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        self.falai = SDXLPlugin()
        self.ctx: agents.JobContext = ctx
        self.video_out = rtc.VideoSource(640, 480)

    async def start(self):
        await self.publish_video()
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)

    def on_data_received(self, data: bytes, participant: rtc.RemoteParticipant, kind):
        payload = json.loads(data.decode("utf-8"))

        if payload["type"] == "user_chat_message":
            text = payload["text"]
            print("Text")

    async def publish_video(self):
        track = rtc.LocalVideoTrack.create_video_track("agent-video", self.video_out)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA
        await self.ctx.room.local_participant.publish_track(track, options)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    async def process_track(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        count = 0
        async for frame in video_stream:
            count += 1
            if count % 10 != 0:
                continue
            frame = await self.falai.generate_image_from_prompt("an image of a cat")
            print("NEIL got frame")
            self.video_out.capture_frame(frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")

        await job_request.accept(
            FalAI.create,
            identity="kitt_agent",
            subscribe_cb=agents.SubscribeCallbacks.VIDEO_ONLY,
            auto_disconnect_cb=agents.AutoDisconnectCallbacks.DEFAULT,
        )

    worker = agents.Worker(
        job_request_cb=job_request_cb, worker_type=agents.JobType.JT_ROOM
    )
    agents.run_app(worker)
