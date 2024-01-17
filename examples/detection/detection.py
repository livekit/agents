# Copyright 2024 LiveKit, Inc.
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

import logging
import time

from livekit import agents, rtc
from livekit.plugins.directai import Detector
from PIL import Image, ImageDraw

INTRO_MESSAGE = """
This example uses DirectAI to detect objects in the video stream. DirectAI allows you to
detect arbitrary objects using prompts instead of needing to know which objects to detect
beforehand. This example detects cell phones by default, but you can change the prompt via
a text chat with the agent. Each comma-separated word is sent in as an "item to include".
For example, "cell-phone, phone, mobile-phone" would be a good prompt to detect cell phones.
"""


class Detection:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        instance = Detection(ctx)
        await instance.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx: agents.JobContext = ctx
        self.detector = Detector(
            detector_configs=[
                Detector.DetectorConfig(
                    name="item",
                    examples_to_include=["cell phone"],
                    examples_to_exclude=[],
                    detection_threshold=0.2,
                )
            ]
        )
        self.video_out = rtc.VideoSource(width=640, height=480)
        self.latest_results = []
        self.detecting = False
        self.chat = rtc.ChatManager(ctx.room)
        self.chat.on("message_received", self.on_chat_received)

    async def start(self):
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        await self.chat.send_message(INTRO_MESSAGE)
        await self.publish_video()

    def on_chat_received(self, message: rtc.ChatMessage):
        text = message.message
        words = text.split(",")
        includes = []
        for word in words:
            word = word.strip()
            if len(word) > 0:
                includes.append(word)
        self.detector = Detector(
            detector_configs=[
                Detector.DetectorConfig(
                    name="item",
                    examples_to_include=includes,
                    examples_to_exclude=[],
                    detection_threshold=0.2,
                )
            ]
        )

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    async def process_track(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        last_processed_time = 0
        frame_interval = 0.5  # 2 fps
        async for frame in video_stream:
            current_time = time.time()
            if (current_time - last_processed_time) >= frame_interval:
                last_processed_time = current_time
                self.ctx.create_task(self.detect(frame))

            if len(self.latest_results) == 0:
                self.video_out.capture_frame(frame)
                continue

            argb_frame = rtc.ArgbFrame.create(
                format=rtc.VideoFormatType.FORMAT_RGBA,
                width=frame.buffer.width,
                height=frame.buffer.height,
            )
            frame.buffer.to_argb(dst=argb_frame)
            image = Image.frombytes(
                "RGBA", (argb_frame.width, argb_frame.height), argb_frame.data
            )

            # Draw red bounding box
            draw = ImageDraw.Draw(image, mode="RGBA")
            for result in self.latest_results:
                draw.rectangle(
                    (result.top_left, result.bottom_right),
                    outline=(255, 0, 0, 255),
                    width=3,
                )

            # LiveKit uses ARGB little-endian (so BGRA big-endian)
            (r, g, b, a) = image.split()
            # PIL we say "RGBA" because that's what PIL supports. But has no consequence, we store as BGRA
            argb_image = Image.merge("RGBA", (b, g, r, a))
            argb_frame = rtc.ArgbFrame.create(
                format=rtc.VideoFormatType.FORMAT_ARGB,
                width=frame.buffer.width,
                height=frame.buffer.height,
            )
            # LiveKit stores underlying data as little-endian. So we set the BGRA data directly to an ARGB frame
            argb_frame.data[:] = argb_image.tobytes()
            result_frame = rtc.VideoFrame(argb_frame.to_i420())
            self.video_out.capture_frame(result_frame)

    async def detect(self, frame: rtc.VideoFrame):
        if self.detecting:
            return

        self.detecting = True
        try:
            results = await self.detector.detect(frame=frame)
            self.latest_results = results
        finally:
            self.detecting = False

    async def publish_video(self):
        track = rtc.LocalVideoTrack.create_video_track("agent-video", self.video_out)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA
        await self.ctx.room.local_participant.publish_track(track, options)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")

        await job_request.accept(
            Detection.create,
            identity="detection_agent",
            auto_subscribe=agents.AudoSubscribe.VIDEO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(
        job_request_cb=job_request_cb, worker_type=agents.JobType.JT_ROOM
    )
    agents.run_app(worker)
