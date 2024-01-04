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

import asyncio
import json
import logging
import time
from enum import Enum

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
        # state
        self.state = StateManager(ctx)

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

    async def start(self):
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("data_received", self.on_data_received)
        await self.send_message_from_agent(INTRO_MESSAGE)
        await self.publish_video()

    def on_data_received(self, data: bytes, participant: rtc.RemoteParticipant, kind):
        payload = json.loads(data.decode("utf-8"))

        if payload["type"] == "user_chat_message":
            text = payload["text"]
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
                format=rtc.VideoFormatType.FORMAT_ARGB,
                width=frame.buffer.width,
                height=frame.buffer.height,
            )
            frame.buffer.to_argb(dst=argb_frame)
            image = Image.frombytes(
                "RGBA", (argb_frame.width, argb_frame.height), argb_frame.data
            )  # Underlying data is BGRA which is unsupported by PIL but is the format LiveKit uses
            draw = ImageDraw.Draw(image)
            for result in self.latest_results:
                draw.rectangle(
                    (result.top_left, result.bottom_right), outline="#ff0000", width=3
                )
            argb_frame = rtc.ArgbFrame.create(
                format=rtc.VideoFormatType.FORMAT_ARGB,
                width=frame.buffer.width,
                height=frame.buffer.height,
            )
            argb_frame.data[:] = image.tobytes()
            result_frame = rtc.VideoFrame(
                0, rtc.VideoRotation.VIDEO_ROTATION_0, argb_frame.to_i420()
            )
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

    async def send_message_from_agent(self, text):
        await self.ctx.room.local_participant.publish_data(
            json.dumps({"type": "agent_chat_message", "text": text})
        )


UserState = Enum("UserState", "SPEAKING, SILENT")
AgentState = Enum("AgentState", "LISTENING, THINKING, SPEAKING")


class StateManager:
    def __init__(self, ctx: agents.JobContext):
        self._agent_sending_audio = False
        self._chat_gpt_working = False
        self._user_state = UserState.SILENT
        self._ctx = ctx

    async def _send_datachannel_message(self):
        msg = json.dumps(
            {
                "type": "state",
                "user_state": self.user_state.name.lower(),
                "agent_state": self.agent_state.name.lower(),
            }
        )
        await self._ctx.room.local_participant.publish_data(msg)

    @property
    def agent_sending_audio(self):
        return self._agent_sending_audio

    @agent_sending_audio.setter
    def agent_sending_audio(self, value):
        self._agent_sending_audio = value
        asyncio.create_task(self._send_datachannel_message())

    @property
    def chat_gpt_working(self):
        return self._chat_gpt_working

    @chat_gpt_working.setter
    def chat_gpt_working(self, value):
        self._chat_gpt_working = value
        asyncio.create_task(self._send_datachannel_message())

    @property
    def user_state(self):
        return self._user_state

    @user_state.setter
    def user_state(self, value):
        self._user_state = value
        asyncio.create_task(self._send_datachannel_message())

    @property
    def agent_state(self):
        if self.agent_sending_audio:
            return AgentState.SPEAKING

        if self.chat_gpt_working:
            return AgentState.THINKING

        return AgentState.LISTENING


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")

        await job_request.accept(
            Detection.create,
            identity="detection_agent",
            subscribe_cb=agents.SubscribeCallbacks.VIDEO_ONLY,
            auto_disconnect_cb=agents.AutoDisconnectCallbacks.DEFAULT,
        )

    worker = agents.Worker(
        job_request_cb=job_request_cb, worker_type=agents.JobType.JT_ROOM
    )
    agents.run_app(worker)
