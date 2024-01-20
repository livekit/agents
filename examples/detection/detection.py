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
from typing import AsyncIterable, Optional

from livekit import agents, rtc
from livekit.plugins.directai import Detector
from livekit.plugins.elevenlabs import TTS
from PIL import Image, ImageDraw

INTRO_MESSAGE = """
Hi there! I can help you detect objects in your video stream using Direct AI's real-time object detector.
Wanna see what I can do? Try typing in "eyes". You can change what objects to detect by typing them in
the chat.
"""

BYE_MESSAGE = """
Thanks for giving this a try! Goodbye for now.
"""

_DETECTION_THRESHOLD = 0.15
_OUTPUT_HEIGHT = 720
_OUTPUT_WIDTH = 960
_ELEVEN_TTS_SAMPLE_RATE = 44100
_ELEVEN_TTS_CHANNELS = 1


class Detection:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        instance = Detection(ctx)
        await instance.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx: agents.JobContext = ctx
        self.detector: Optional[Detector] = None
        self.tts_plugin = TTS(model_id="eleven_turbo_v2")
        self.video_out = rtc.VideoSource(_OUTPUT_WIDTH, _OUTPUT_HEIGHT)
        self.audio_out = rtc.AudioSource(_ELEVEN_TTS_SAMPLE_RATE, _ELEVEN_TTS_CHANNELS)
        self.latest_results = []
        self.detecting = False
        self.chat = rtc.ChatManager(ctx.room)
        self.chat.on("message_received", self.on_chat_received)

    async def start(self):
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            self.ctx.create_task(self.process_track(track))

        self.ctx.room.on("track_subscribed", on_track_subscribed)

        video_track = rtc.LocalVideoTrack.create_video_track(
            "agent-video", self.video_out
        )
        await self.ctx.room.local_participant.publish_track(video_track)

        audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-mic", self.audio_out
        )
        await self.ctx.room.local_participant.publish_track(audio_track)
        self.tts_stream = self.tts_plugin.stream()
        self.ctx.create_task(self.send_audio_stream(self.tts_stream))

        # give time for the subscriber to fully subscribe to the agent's tracks
        await asyncio.sleep(1)
        await self.send_chat_and_voice(INTRO_MESSAGE)

        # limit to 2 mins
        self.ctx.create_task(self.end_session_after(2 * 60))

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
                    detection_threshold=_DETECTION_THRESHOLD,
                )
            ]
        )

    async def end_session_after(self, duration: int):
        await asyncio.sleep(duration)
        await self.send_chat_and_voice(BYE_MESSAGE)
        await asyncio.sleep(5)
        await self.ctx.disconnect()

    async def send_chat_and_voice(self, message: str):
        self.tts_stream.push_text(message)
        await self.chat.send_message(message)

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
        if (not self.detector) or self.detecting:
            return

        self.detecting = True
        try:
            results = await self.detector.detect(frame=frame)
            self.latest_results = results
        finally:
            self.detecting = False

    async def send_audio_stream(
        self, tts_stream: AsyncIterable[agents.tts.SynthesisEvent]
    ):
        async for e in tts_stream:
            if e.type == agents.tts.SynthesisEventType.AUDIO:
                await self.audio_out.capture_frame(e.audio.data)
        await tts_stream.aclose()

    def update_state(self, state: str):
        metadata = json.dumps({"agent_state": state})
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for Detection")

        await job_request.accept(
            Detection.create,
            identity="detection_agent",
            auto_subscribe=agents.AutoSubscribe.VIDEO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(job_request_cb)
    agents.run_app(worker)
